from logging import root
from random import sample
# from model.pointnet2 import PointCloudNorm
from model import get_model
from model.model_utils import get_loss, add_sin_difference
from dataset.kitti_dataset import KittiDataset
import argparse
from utils.train_utils import *
import datetime
import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader
import numpy as np
from scipy import stats

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--output_dir', type=str, default='./output', help='output saving path')
    parser.add_argument('--batch_size_per_gpu', type=int, default=8, help='batch size for distributed training')
    parser.add_argument('--pretrained_model', type=str, default='None', help='pretrained model')
    parser.add_argument('--width_threshold', type=float, default=20.0, help='width threshold for filtering')
    parser.add_argument('--depth_threshold', type=float, default=20.0, help='depth threshold for filtering')
    parser.add_argument('--npoints', type=int, default=30000, help='sampled input point cloud num percentage')
    parser.add_argument('--model_name', type=str, default='pointnet_rot_dir', help='indicate the specific model')
    parser.add_argument('--model_tag', type=str, default='pointnet_rot_dir', help='model tag for the current model')
    parser.add_argument('--max_threshold', type=float, default=1.0, help='select cloud sequence whose rotation is larger than this threshold')
    parser.add_argument('--min_threshold', type=float, default=1.0, help='select cloud sequence whose rotation is smaller than this threshold')
    parser.add_argument('--pkl_file', type=str, default=None, help='cloud pkl file path')
    parser.add_argument('--to_file', action='store_true', default=False, help='write abnormal cloud to file')
    parser.add_argument('--aug_label_path', type=str, default='./data/augmentation_label.npy', help='data augmentation label')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_config()
    if opt.launcher == 'slurm':
        total_gpus, opt.local_rank = init_dist_slurm(opt.tcp_port, opt.local_rank, backend='nccl')

    opt.output_dir = opt.output_dir + '/' + opt.model_tag
    ckpt_dir = opt.output_dir + '/ckpt'
    if not os.path.exists(ckpt_dir):
        os.system('mkdir -p %s' % (ckpt_dir))

    log_file = opt.output_dir + '/log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = create_logger(log_file, rank=opt.local_rank)

    logger.info('***************Start Initializing model*****************')
    # initialize model
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(opt).items():
        logger.info('{:16} {}'.format(key, val))

    # Initialize model
    model = get_model(opt.model_name).cuda()

    start_epoch = 0
    # TODO: Implement continue training
    logger.info('Loading pretrained model from %s' % (opt.pretrained_model))
    state_params = torch.load(opt.pretrained_model, map_location=torch.device('cpu'))
    model_state = state_params['model_state']
    model.load_state_dict(model_state)
    logger.info('succeed in loading model')
    logger.info(model)
    # launch trainer
    model.eval()
    # Data distributed training
    if opt.launcher == 'slurm':
        logger.info('total batch size: %d ' % (total_gpus * opt.batch_size_per_gpu))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank % torch.cuda.device_count()])

    # Dataparallel training
    elif len(opt.device_ids) > 1:
        logger.info('Dataparallel training')
        device_ids = [int(item) for item in opt.device_ids.split(',')]
        model = nn.parallel.DataParallel(model, device_ids=device_ids)

    # loading dataset
    test_set = KittiDataset(info_path=opt.pkl_file, logger=logger, opt=opt)
    logger.info('testing file length is %d' % (len(test_set)))

    if opt.launcher == 'slurm':
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=opt.batch_size_per_gpu, shuffle=False, drop_last=False, \
                                    pin_memory=True, num_workers=opt.workers, sampler=test_sampler)
    else:
        test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, \
                                    pin_memory=True)

    logger.info('********Start Evaluating***********')
    rot_result = np.array([])
    dir_result = np.array([])
    pbar = tqdm.tqdm(total=len(test_loader), leave=False, desc='eval')
    index_list = []
    if opt.to_file:
        f_abnormal = open('%s/abnormal.txt' % (opt.output_dir), 'w')
        point_info = test_set._get_point_info()
    for i, data in enumerate(test_loader):
        input_points = data['points'].cuda()
        rot_pred, dir_pred = model(input_points)
        negative_index = dir_pred < 0.5
        dir_pred[negative_index] = -1
        dir_pred[~negative_index] = 1
        rot_pred = (rot_pred / 3.14 * 180).squeeze(dim=-1).data.cpu().numpy()
        dir_pred = dir_pred.squeeze(dim=-1).data.cpu().numpy()
        dir_result = np.concatenate((dir_result, dir_pred), axis=0)
        rot_result = np.concatenate((rot_result, rot_pred), axis=0)
        if opt.to_file:
            error_index = np.where(rot_pred < opt.min_threshold)[0]
            error_index = np.concatenate((error_index, np.where(rot_pred > opt.max_threshold)[0]), axis=0)
            for i in list(error_index):
                f_abnormal.write(point_info[int(data['sample_index'][i].data.cpu())]['velodyne_path'] + ' %.2f' %rot_pred[i] + '\n')
        if opt.local_rank == 0:
            pbar.update()
    dir_mod = stats.mode(dir_result)[0]
    rot_result = rot_result.astype(np.int64) * dir_mod

    logger.info('finished prediction')
    rot_mean = rot_result.mean()
    rot_mid = np.median(rot_result)
    logger.info('avg rot is %.2f' % rot_mean)
    logger.info('avg rot mid is %d' % rot_mid)
    logger.info('max and mix is %d %d' %(rot_result.max(), np.abs(rot_result).min()))
    mode = stats.mode(rot_result)
    rot_mode = mode[0]
    logger.info('mode is %d' % mode[0])
    if opt.to_file:
        f_abnormal.close()
    # a hard code here
    for i in range(5):
        rot_result = np.delete(rot_result, np.argmax(rot_result))
        rot_result = np.delete(rot_result, np.argmin(rot_result))
    rot_mean_filter = rot_result.mean()
    logger.info('rot_mean  filtered is %.2f' % rot_mean_filter)


if __name__ == '__main__':
    main()

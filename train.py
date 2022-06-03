from logging import root
from random import sample
from model import get_model
#from dataset import get_dataset
from model.model_utils import get_loss, add_sin_difference
from dataset.kitti_dataset import KittiDataset
from dataset.data_list import TRAIN_INFO_PATH
import argparse
from utils.train_utils import *
import datetime
import tqdm
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, dataloader


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--epochs', type=int, default=30, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--output_dir', type=str, default='./output', help='output saving path')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--LR', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--lr_schedule', type=str, default='None', help='indicate lr schedule')
    parser.add_argument('--scheduler_param', type=str, default='', help='specific scheduler parameters')
    parser.add_argument('--BETA', type=float, default=0.999, help='adam optimizer parameter')
    parser.add_argument('--L1_WEIGHT', type=float, default=3.0, help='L1 loss weight')
    parser.add_argument('--batch_size_per_gpu', type=int, default=8, help='batch size for distributed training')
    parser.add_argument('--model_tag', type=str, default='baseline', help='inidicate the training model name')
    parser.add_argument('--device_ids', type=str, default='0', help='indicates CUDA devices, e.g. 0,1,2')
    parser.add_argument('--pretrained_model', type=str, default='None', help='pretrained model')
    parser.add_argument('--width_threshold', type=float, default=20.0, help='width threshold for filtering')
    parser.add_argument('--depth_threshold', type=float, default=20.0, help='depth threshold for filtering')
    parser.add_argument('--npoints', type=int, default=30000, help='sample input point cloud num percentage')
    parser.add_argument('--model_name', type=str, default='pointnet_rot_dir', help='indicate the specific model')
    parser.add_argument('--loss_fn', type=str, default='L1', help='specify loss function')
    parser.add_argument('--rot_weight', type=float, default=10.0, help='rot loss weight')
    parser.add_argument('--shift_weight', type=float, default=1.0, help='shift loss weight')
    parser.add_argument('--aug_label_path', type=str, default='./data/augmentation_label.npy', help='data augmentation label')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_config()

    if opt.lr_schedule is not None:
        opt.scheduler_param = json.loads(opt.scheduler_param)

    dist_train = False
    # define the launcher, the distributed training is only utilized when slurm launcher is applied
    if opt.launcher == 'slurm':
        # 使用slurm启动并行的训练的流程
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
    # initialize optimizer
    optim = create_optimizer(model, opt)
    # initialize lr_scheduler
    scheduler = create_schedule(optim, opt)

    start_epoch = 0
    # TODO: Implement continue training
    if opt.pretrained_model != 'None':
        logger.info('Loading pretrained model from %s' % (opt.pretrained_model))
        state_params = torch.load(opt.pretrained_model, map_location=torch.device('cpu'))
        model_state = state_params['model_state']
        model.load_state_dict(model_state)
        optim_state = state_params['optimizer_state']
        optim.load_state_dict(optim_state)
        start_epoch = state_params['epoch'] + 1
        logger.info('succeed in loading model')
    else:
        logger.info('Training from scratch')

    logger.info(model)

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
    train_set = KittiDataset(info_path=TRAIN_INFO_PATH, logger=logger, opt=opt, training=True)
    # launch trainer
    model.train()
    if opt.launcher == 'slurm':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)

        train_loader = DataLoader(train_set, batch_size=opt.batch_size_per_gpu, shuffle=False, drop_last=True, \
                                    pin_memory=True, num_workers=opt.workers, sampler=train_sampler)

    else:
        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, \
                                    pin_memory=True)

    logger.info('********Start Training***********')
    for epoch in range(start_epoch, opt.epochs):
        for i, data in enumerate(train_loader):
            input_points = data['points'].cuda()
            rot_label = data['rotation'].to(torch.float32).cuda().unsqueeze(dim=-1)
            dir_label = data['dir_targets'].to(torch.float32).cuda().unsqueeze(dim=-1)
            rot_pred, dir_pred = model(input_points)
            inverse_mask = rot_label < 0
            rot_label[inverse_mask] = rot_label[inverse_mask] * -1
            pred_embedding, target_embedding = add_sin_difference(rot_pred, rot_label)
            rot_loss = get_loss(pred_embedding, target_embedding, opt.loss_fn) * opt.rot_weight
            dir_loss = get_loss(dir_pred, dir_label, 'binary')
            loss = rot_loss + dir_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            logger.info('Epoch %d iter %d/%d : rot_loss %.2f dir_loss %.2f' % (epoch, i, len(train_loader), \
                        rot_loss.data.cpu().numpy(), dir_loss.data.cpu().numpy()))
        if opt.local_rank == 0 and (epoch+1) % opt.ckpt_save_interval == 0 :
           save_model(model, ckpt_dir, optim, epoch)
        if scheduler is not None:
            scheduler.step()

if __name__ == '__main__':
    main()

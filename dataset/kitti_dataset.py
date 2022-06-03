import os
from collections import defaultdict
import torch.utils.data as torch_data
import torch.nn.functional as F
import numba
import io
import pickle
import numpy as np
from dataset.data_utils import *

def pkl_load(file_path):
    read_data = open(file_path, 'rb')
    return pickle.load(read_data)

def pc_normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / m
    return points

def txt_file(txt_file):
    with open(txt_file, 'r') as f:
        velodyne_path = f.readlines()
    return velodyne_path

class KittiDataset(torch_data.Dataset):
    def __init__(self, info_path, logger=None, opt=None, training=False):
        super().__init__()
        self.depth_threshold = opt.depth_threshold
        self.npoints = opt.npoints
        self.logger = logger
        self.width_threshold = opt.width_threshold
        self.opt = opt
        self.training = training

        # if info_path is not None, load kitti info from pkl files
        if info_path.endswith('pkl'):
            try:
                self.logger.info('Loading data from pkl')
                infos_list = info_path.split(';')
                self.kitti_infos = pkl_load(infos_list[0].strip())
                if len(infos_list) > 1:
                    for idx in range(1, len(infos_list), 1):
                        infos = pkl_load(infos_list[idx].strip())
                        self.kitti_infos.extend(infos)
                # append the cloud path to annos
                for i in range(len(self.kitti_infos)):
                    frame_cloud_path = self.kitti_infos[i]['velodyne_path']
                frame_id_array = np.array([frame_cloud_path])
                #self.kitti_infos[i]['annos']['image_id'] = frame_id_array
            except:
                raise Exception('No data info found')
        else:
            infos_list = info_path.split(';')
            self.kitti_infos = txt_file(infos_list[0].strip())
            if len(infos_list) > 1:
                for idx in range(1, len(infos_list), 1):
                    self.kitti_infos.extend(txt_file(infos_list[i].strip()))

        if not self.training:
            self.aug_label = np.load(self.opt.aug_label_path)

    def get_lidar(self, idx, velodyne_path=''):
        if velodyne_path[0] != '/':
            lidar_file = os.path.join(self.kitti_split_path, 'velodyne', '%06d.bin' % idx)
        else:
            lidar_file = velodyne_path

        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        max_point = 0
        for cur_sample in batch_list:
            cur_num = cur_sample['points'].shape[0]
            if cur_num > max_point:
                max_point = cur_num
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            point_list = []
            if key in ['points']:
                for point in data_dict[key]:
                    padding_zero = np.zeros((max_point - point.shape[0], 4))
                    point = np.concatenate((point, padding_zero), axis=0).astype(np.float32)
                    point_list.append(point)
                ret[key] = np.stack(point_list, axis=0)
                ret[key] = torch.from_numpy(ret[key])
            elif key in ['dir_targets', 'rotation']:
                ret[key] = np.stack(data_dict[key], axis=0)
                ret[key] = torch.from_numpy(ret[key])
        return ret


    def __len__(self):
        return len(self.kitti_infos)

    def _get_point_info(self):
        return self.kitti_infos

    def __getitem__(self, index):
        if isinstance(self.kitti_infos[index], dict):
            info = self.kitti_infos[index]
            info['image_idx'] = index
            sample_idx = info['image_idx']

            velodyne_path = info['velodyne_path']
            points = self.get_lidar(sample_idx, velodyne_path=velodyne_path)
            # select points with width threshold
        else:
            velodyne_path = self.kitti_infos[index].strip()
            sample_idx = index
            points = self.get_lidar(sample_idx, velodyne_path=velodyne_path)

        if self.training:
            # global rotation
            points, rotation = global_rotation(points, rotation=np.pi/2)

            if rotation > 0:
                dir_targets = 1.0
            else:
                dir_targets = 0.
            points = points_width_filter(points, threshold=self.width_threshold)
            points = points_depth_filter(points, threshold=self.depth_threshold)
            point_num = points.shape[0]
            if self.npoints > point_num:
                padding_num = self.npoints - point_num
                padding_point = np.zeros((padding_num, 4))
                points = np.concatenate((points, padding_point), axis=0)
            points = get_topk_points(points, topk=self.npoints, dim=2, inverse=True)

        else:

#            rotation, z_shift = self.aug_label[index][0], self.aug_label[index][1]
#            if self.opt.rot_aug:
#                self.logger.info('rotation')
#                points[:, :3] = rotation_points_single_angle(points[:, :3], rotation, axis=2)
#            if rotation > 0:
#                dir_targets = 1.0
#            else:
#                dir_targets = 0.0
            rotation = 0.
            dir_targets = 0.
            point_num = points.shape[0]
            points = points_width_filter(points, threshold=self.width_threshold)
            points = points_depth_filter(points, threshold=self.depth_threshold)
            point_num = points.shape[0]
            if self.npoints > point_num:
                padding_num = self.npoints - point_num
                padding_point = np.zeros((padding_num, 4))
                points = np.concatenate((points, padding_point), axis=0)
            points = get_topk_points(points, topk=self.npoints, dim=2, inverse=True)

        input_dict = {
            'points': points,
            'rotation': rotation,
            'dir_targets': dir_targets,
            'sample_index': index
        }

        return input_dict

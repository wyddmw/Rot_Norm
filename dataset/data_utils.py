import numpy as np
import numba
import torch

def global_rotation(points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    return points, noise_rotation

def global_rotation_single(points, rotation=np.pi / 4):
    points[:, :3] = rotation_points_single_angle(
        points[:, :3], rotation, axis=2)
    return points, rotation

def global_rotation_nodir(points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    if noise_rotation == -np.pi/2:
        noise_rotation = np.pi / 2
    points[:, :3] = rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    return points, noise_rotation

def global_shift(points, z_shift=15.):
    if not isinstance(z_shift, list):
        shift = [-z_shift, z_shift]
    noise_shift = np.random.uniform(shift[0], shift[1])
    points[:, 2] = points[:, 2] + noise_shift
    return points, noise_shift

@numba.jit(nopython=True)
def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [rot_cos, 0, -rot_sin, 0, 1, 0, rot_sin, 0, rot_cos],
            dtype=points.dtype).reshape(3, 3)
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [rot_cos, -rot_sin, 0, rot_sin, rot_cos, 0, 0, 0, 1],
            dtype=points.dtype).reshape(3, 3)
    elif axis == 0:
        rot_mat_T = np.array(
            [1, 0, 0, 0, rot_cos, -rot_sin, 0, rot_sin, rot_cos],
            dtype=points.dtype).reshape(3, 3)
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T

def get_topk_points(points, topk=2000, dim=2, inverse=False):
    if not isinstance(points, torch.Tensor):
        points = torch.FloatTensor(points)
    points_value = points[:, dim]
    if inverse:
        points_value = points_value * -1
    topk_value, topk_index = torch.topk(points_value, k=topk)
    selected_mean = (topk_value * -1).mean()
    points_selected = points[topk_index].numpy()

    return points_selected

def points_width_filter(points, threshold=40):
    points_value = points[:, 1]
    points_width_flag = (points_value < threshold) & (points_value > (threshold * -1))
    points_selected = points[points_width_flag]
    return points_selected

def points_depth_filter(points, threshold=40):
    points_value = points[:, 0]
    points_width_flag = (points_value < threshold) & (points_value > (threshold * -1))
    points_selected = points[points_width_flag]
    return points_selected

def select_points(points, npoint=16384, depth_threshold=40.0):
        if points.shape[0] < npoint:
            # current points are less than required and padding is added
            choice = np.arange(0, points.shape[0], dtype=np.int32)
            extra_choice = np.random.choice(choice, npoint-points.shape[0], replace=False)      # 是否进行重复采样
            choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        elif points.shape[0] > npoint:
            # current points are more than required
            # first filter out by depth threshold
            pts_depth = points[:, 2]
            pts_near_flag = pts_depth < depth_threshold
            far_idxs = np.where(pts_near_flag==0)[0]
            near_idxs = np.where(pts_near_flag==1)[0]

            if len(near_idxs) < npoint:
                near_idxs_choice = near_idxs
                far_idxs_choice = np.random.choice(far_idxs, npoint-len(near_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs, far_idxs_choice), aixs=0)
            else:
                choice = np.random.choice(near_idxs, npoint, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, points.shape[0], dtype=np.int32)

        points = points[choice, :]
        return points

def generate_testing_targets(data_length=12436, rotation_range=np.pi/2, shift_range=15.0, save_dir='./data/augmentation_label'):
    augmentation_label = np.zeros((data_length, 2))
    if not isinstance(rotation_range, list):
        rotation = [-rotation_range, rotation_range]
    if not isinstance(shift_range, list):
        shift = [-shift_range, shift_range]
    for i in range(data_length):
        # shift = [-z_shift, z_shift]
        augmentation_label[i][1] = np.random.uniform(shift[0], shift[1])
        augmentation_label[i][0] = np.random.uniform(rotation[0], rotation[1])

    np.save(save_dir, augmentation_label)


if __name__ == '__main__':
    generate_testing_targets()

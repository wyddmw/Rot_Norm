from model.pointnet2_rot_dir import PointCloudNormRotDir
from model.pointnet2_rot_dir_double import PointCloudNormRotDirDouble
from model.pointnet_rot_dir import PointNetRotDir
from model.pointnet_rot_dir_double import PointNetRotDirDouble


def get_model(model_name):
    if model_name == 'point2_rot_dir':
        model = PointCloudNormRotDir()
    elif model_name == 'point2_rot_dir_double':
        model = PointCloudNormRotDirDouble()
    elif model_name == 'pointnet_rot_dir':
        model = PointNetRotDir()
    elif model_name == 'pointnet_rot_dir_double':
        model = PointNetRotDirDouble()
    else:
        raise NotImplemented('Wrong Model')
    return model

def add_sin_difference(rot_pred, rot_target):
    # sin(A - B) = sinAcosB - cosAsinB
    rot_pred_embedding = torch.sin(rot_pred) * torch.cos(rot_target)
    rot_target_embedding = torch.cos(rot_pred) * torch.sin(rot_target)
    return rot_pred_embedding, rot_target_embedding


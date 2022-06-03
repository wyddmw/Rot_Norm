import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_loss(pred, target, loss_type):
    if loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'smoothL1':
        loss = F.smooth_l1_loss(pred, target)
    elif loss_type == 'binary':
        loss = F.binary_cross_entropy(pred, target)
    else:
        raise NotImplemented('wrong loss type')

    return loss

def add_sin_difference(rot_pred, rot_target):
    pred_embedding = torch.sin(rot_pred) * torch.cos(rot_target)
    target_embedding = torch.cos(rot_pred) * torch.sin(rot_target)
    return pred_embedding, target_embedding

def point_adjust(point_cloud, rot, shift):
    pass

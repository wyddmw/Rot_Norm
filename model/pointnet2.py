import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from model.pointnet2_utils import PointNetSetAbstraction


class PointCloudNormAll(nn.Module):
    def __init__(self,in_channel=3):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + in_channel, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + in_channel, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        
        self.rotation = nn.Linear(256, 1)
        self.shift = nn.Linear(256, 1)
        # initialize parameters
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, points):
        B, _, _ = points.shape
        points = points[:, :, :3].transpose(1, 2)       # [N, 3, npoints]

        norm = None
        l1_xyz, l1_points = self.sa1(points, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(F.relu(self.bn2(self.fc2(x))))
        rotation = self.rotation(x)
        shift = self.shift(x)

        return rotation, shift


if __name__ == '__main__':

    fake_input = torch.randn(4, 3, 20000)
    print("Input point shape is: ", fake_input.shape)
    
    model = PointCloudNorm()
    rotation, shift = model(fake_input)

    # initialize optimizer
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    rotation_label = torch.randn(4, 1)
    shift_label = torch.randn(4, 1)
    
    rotation_loss = model.get_rotation_loss(rotation, rotation_label)
    shift_loss = model.get_shift_loss(shift, shift_label)

    pdb.set_trace()


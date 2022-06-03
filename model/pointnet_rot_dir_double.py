import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model.pointnet_utils import PointNetEncoderDouble

class PointNetRotDirDouble(nn.Module):
    def __init__(self, in_channel=3, max_feature=2048):
        super(PointNetRotDirDouble, self).__init__()
        self.feat = PointNetEncoderDouble(max_feature=2048)
        self.fc1 = nn.Linear(max_feature, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.rotation = nn.Linear(256, 1)
        self.dir = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)[:, :3, :]
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        rotation = self.rotation(x)
        direction = self.dir(x)
        direction = self.sigmoid(direction)
        return rotation, direction

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    input = torch.randn(8, 3, 20000)
    model = PointNetRotDirDouble(in_channel=3)
    output = model(input)
    print(output[0].shape)

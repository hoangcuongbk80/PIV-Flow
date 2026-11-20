import torch
from torch import nn


class MotionEncoder(nn.Module):
    def __init__(self, width, corr_channels=324):
        super().__init__()
        self.corr = nn.Sequential(nn.Conv2d(corr_channels, width, 1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.flow = nn.Sequential(nn.Conv2d(2, 64, 7, padding=3), nn.ReLU(),
                                  nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.fuse = nn.Sequential(nn.Conv2d(width + 32, width - 2, 3, padding=1), nn.ReLU())

    def forward(self, corr, flow):
        return torch.cat((self.fuse(torch.cat((self.corr(corr), self.flow(flow)), 1)), flow), 1)


class ConvGRU(nn.Module):
    def __init__(self, hidden, inputs):
        super().__init__()
        self.z = nn.Conv2d(hidden + inputs, hidden, 3, padding=1)
        self.r = nn.Conv2d(hidden + inputs, hidden, 3, padding=1)
        self.q = nn.Conv2d(hidden + inputs, hidden, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat((h, x), 1)
        z, r = self.z(hx).sigmoid(), self.r(hx).sigmoid()
        q = self.q(torch.cat((r * h, x), 1)).tanh()
        return (1 - z) * h + z * q


def prediction_head(width, outputs):
    return nn.Sequential(nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(width * 2, outputs, 1))

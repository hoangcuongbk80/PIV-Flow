"""ResNet BasicBlocks with output stride eight and optional ImageNet weights."""
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if cin != cout or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                            nn.BatchNorm2d(cout))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + identity)


class Encoder(nn.Module):
    def __init__(self, depth, input_channels, output_channels, pretrained=False):
        super().__init__()
        counts = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}[depth]
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin = 64
        for i, (cout, count, stride, dil) in enumerate(zip([64, 128, 256, 512], counts,
                                                          [1, 2, 1, 1], [1, 1, 2, 4]), 1):
            blocks = [BasicBlock(cin, cout, stride, dil)]
            blocks += [BasicBlock(cout, cout, 1, dil) for _ in range(count - 1)]
            setattr(self, f"layer{i}", nn.Sequential(*blocks))
            cin = cout
        self.projection = nn.Conv2d(512, output_channels, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if pretrained:
            from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
            network = (resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if depth == 18
                       else resnet34(weights=ResNet34_Weights.IMAGENET1K_V1))
            state = network.state_dict()
            state.pop("fc.weight")
            state.pop("fc.bias")
            if input_channels == 6:
                state["conv1.weight"] = state["conv1.weight"].repeat(1, 2, 1, 1) / 2
            self.load_state_dict(state, strict=False)
        self.freeze_bn()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_bn()
        return self

    def forward(self, image):
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
        return self.projection(x)

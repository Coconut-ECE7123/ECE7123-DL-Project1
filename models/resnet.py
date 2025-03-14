'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# Suggested Architectural Hyper-params:
# Ci: channel number controlled by "expansion"
# Fi: filter size (3x3Conv) added by (1x1Conv)
# Ki: kernel size in skip connection fixed -> 1x1Conv
# P: pool size in final pooling layer fixed -> 4x4Pool

class BasicBlock(nn.Module):
    # 3x3 Conv(stride=1or2) + BN.ReLU + 3x3 Conv(stride=1) + BN + Res + ReLU
    # expansion = 1 # expansion: input channel num of next stage equals to output channel num of last stage
    # meaning no subsampling or conv interlayer
    def __init__(self, in_planes, planes, expansion, stride=1):
    # in_plane: input channel; planes: output channel; stride: stage control

        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
        # stride = 2 means depth * 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    # expansion = 4

    def __init__(self, in_planes, planes, expansion, stride=1):
    # in_planes: input channel; planes: desired output channel / 4
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, expansion, init_channel=64, num_classes=10):
        super(ResNet, self).__init__()
        # Input layer
        self.in_planes = init_channel # initial channel width

        self.conv1 = nn.Conv2d(3, init_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)  # input layer config.
        self.bn1 = nn.BatchNorm2d(init_channel) # input layer bn.
        # layer ->-> STAGE
        # following architecture default: 1 -> 2  -> 2  -> 2
        # corresponding:    32] input -> 32 -> 16 -> 8  -> 4
        self.layer1 = self._make_layer(block, init_channel, num_blocks[0], expansion, stride=1) # Stage1
        self.layer2 = self._make_layer(block, init_channel*2, num_blocks[1], expansion, stride=2) # Stage2
        self.layer3 = self._make_layer(block, init_channel*4, num_blocks[2], expansion, stride=2) # Stage3
        self.layer4 = self._make_layer(block, init_channel*8, num_blocks[3], expansion, stride=2) # Stage4
        self.linear = nn.Linear(init_channel*8*expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, expansion, stride): # STAGE constructor
        # planes -> output channel number.
        strides = [stride] + [1]*(num_blocks-1) # form stride list
        layers = []
        for stride in strides: # construction process
            # parameter inputs config. are the same: input num + output num + stride
            layers.append(block(self.in_planes, planes, expansion, stride))
            self.in_planes = planes * expansion # automatic input channel number calculation for next layer
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Baseline Experiment:
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], expansion=1, init_channel=64)

# Resnet Bottleneck Expansion:
def ResNet_Bottleneck():
    return ResNet(Bottleneck, [2, 4, 6, 2], expansion=4, init_channel=32)

# Resnet Bottleneck - Group Conv



def test():
    net = ResNet_Bottleneck()
    print(net(torch.randn(1, 3, 32, 32)).size())
    summary(net, input_size=(1, 3, 32, 32))

# test()

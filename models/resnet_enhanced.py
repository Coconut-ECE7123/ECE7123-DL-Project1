import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class SEBlock(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_planes, in_planes // reduction)
        self.fc2 = nn.Linear(in_planes // reduction, in_planes)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

def activation(input, type):
    if type == 0:
        return F.relu(input)
    elif type == 1:
        return F.leaky_relu(input, negative_slope=0.01)

class Bottleneck_v2(nn.Module):
    def __init__(self, in_planes, planes, expansion, leaky, SE, G, stride=1, reduction=16, scale=1):
        super(Bottleneck_v2, self).__init__()
        self.type = leaky
        self.scale = scale
        self.expansion = expansion
        self.g_conv = G
        self.se = SE
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if G:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
            self.pointwise = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        if SE:
            self.se_block = SEBlock(self.expansion * planes, reduction)
        else:
            self.se_block = None

    def forward(self, x):
        out = activation(self.bn1(self.conv1(x)), self.type)
        if self.g_conv:
            out = activation(self.bn2(self.pointwise(self.conv2(out))), self.type)
        else:
            out = activation(self.bn2(self.conv2(out)), self.type)

        out = self.bn3(self.conv3(out))

        out += self.scale * self.shortcut(x)

        if self.se_block is not None:
            out = self.se_block(out)

        out = activation(out, self.type)
        return out


class ResNet_v2(nn.Module):
    def __init__(self, block, num_blocks, act_config, SE_config, G_config, expansion, init_channel=64, num_classes=10):
        super(ResNet_v2, self).__init__()
        # Input layer
        self.in_planes = init_channel # initial channel width

        self.conv1 = nn.Conv2d(3, init_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channel)
        self.conv2 = nn.Conv2d(init_channel, init_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(init_channel)
        self.layer1 = self._make_layer(block, init_channel, num_blocks[0], expansion,
                                       act_config[0], SE_config[0], G_config[0], stride=1) # Stage1
        self.layer2 = self._make_layer(block, init_channel*2, num_blocks[1], expansion,
                                       act_config[1], SE_config[1], G_config[1], stride=2) # Stage2
        self.layer3 = self._make_layer(block, init_channel*4, num_blocks[2], expansion,
                                       act_config[2], SE_config[2], G_config[2], stride=2) # Stage3
        self.layer4 = self._make_layer(block, init_channel*8, num_blocks[3], expansion,
                                       act_config[3], SE_config[3], G_config[3],stride=2) # Stage4
        self.linear = nn.Linear(init_channel*8*expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, expansion, leaky, SE, G, stride): # STAGE constructor
        # planes -> output channel number.
        strides = [stride] + [1]*(num_blocks-1) # form stride list
        layers = []
        for stride in strides: # construction process
            # parameter inputs config. are the same: input num + output num + stride
            layers.append(block(self.in_planes, planes, expansion, leaky, SE, G, stride))
            self.in_planes = planes * expansion # automatic input channel number calculation for next layer
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Resnet_Leaky():
    return ResNet_v2(Bottleneck_v2, [2,4,6,2], [1,1,1,1],
                     [0,0,0,0], [0,0,0,0], expansion=4, init_channel=32)

def Resnet_SE():
    return ResNet_v2(Bottleneck_v2, [2,4,6,2], [0,0,0,0],
                     [0,0,1,1], [0,0,0,0], expansion=4, init_channel=32)

def Resnet_G():
    return ResNet_v2(Bottleneck_v2, [3,4,6,3], [0,0,0,0],
                     [0,0,0,0], [0,0,1,1], expansion=4, init_channel=32)

def Resnet_Custom():
    return ResNet_v2(Bottleneck_v2, [2, 4, 5, 2], [0, 0, 1, 1],
                     [0, 0, 1, 1], [0, 0, 0, 0], expansion=4, init_channel=32)

def test():
    net = Resnet_Custom()
    print(net(torch.randn(1, 3, 32, 32)).size())
    summary(net, input_size=(1, 3, 32, 32))

# test()












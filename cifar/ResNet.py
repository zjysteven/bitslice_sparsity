from __future__ import print_function

import math

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ["resnet"]

BITWIDTH = 8
NEW_BITWIDTH = 6


def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {
        n: {
            "method": torch.autograd.Variable(
                torch.IntTensor(np.array([method])), requires_grad=False
            ),
            "scale": torch.autograd.Variable(
                torch.IntTensor(np.array([scale])), requires_grad=False
            ),
            "bitwidth": torch.autograd.Variable(
                torch.IntTensor(np.array([bitwidth])), requires_grad=False
            ),
        }
        for n in names
    }


#################################### ugly version ####################################
def conv3x3(in_planes, out_planes, nf_fix_params, stride=1):
    kwargs = {'kernel_size': 3, 'stride': stride, 'padding': 1, 'bias': False}
    "3x3 convolution with padding"
    return nnf.Conv2d_fix(
            in_planes, out_planes, nf_fix_params=nf_fix_params, **kwargs
        )

    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=1, bias=False)


class BasicBlock(nnf.FixTopModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride

        # initialize some fix configurations
        self.conv1_fix_params = _generate_default_fix_cfg(
            ["weight"], method=1, bitwidth=BITWIDTH)
        self.conv2_fix_params = _generate_default_fix_cfg(
            ["weight"], method=1, bitwidth=BITWIDTH)
        '''
        self.bn1_fix_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1, bitwidth=BITWIDTH,
        )
        self.bn2_fix_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1, bitwidth=BITWIDTH,
        )
        '''
        activation_num = 7 if not downsample else 8
        self.fix_params = [
            _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
            for _ in range(activation_num)
        ]

        # initialize layers with corresponding configurations
        self.conv1 = conv3x3(inplanes, planes, self.conv1_fix_params, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.bn1 = nnf.BatchNorm2d_fix(planes, nf_fix_params=self.bn1_fix_params)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, self.conv2_fix_params)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.bn2 = nnf.BatchNorm2d_fix(planes, nf_fix_params=self.bn1_fix_params)

        # initialize activation fix modules
        for i in range(len(self.fix_params)):
            setattr(self, "fix"+str(i), nnf.Activation_fix(nf_fix_params=self.fix_params[i]))

        # initialize weights
        for m in self.modules():
            if isinstance(m, nnf.Conv2d_fix):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                #m.bias.data.zero_()

    def forward(self, x):
        x = self.fix0(x)
        residual = x

        out = self.fix1(self.conv1(x))
        out = self.fix2(self.bn1(out))
        out = self.relu(self.fix3(out))

        out = self.fix4(self.conv2(out))
        out = self.fix5(self.bn2(out))

        if self.downsample is not None:
            residual = self.fix7(self.downsample(x))

        out += residual
        out = self.relu(self.fix6(out))

        return out


class Bottleneck(nnf.FixTopModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride

        # initialize some fix configurations
        self.conv1_fix_params = _generate_default_fix_cfg(
            ["weight"], method=1, bitwidth=BITWIDTH)
        self.conv2_fix_params = _generate_default_fix_cfg(
            ["weight"], method=1, bitwidth=BITWIDTH)
        self.conv3_fix_params = _generate_default_fix_cfg(
            ["weight"], method=1, bitwidth=BITWIDTH)
        '''
        self.bn1_fix_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1, bitwidth=BITWIDTH,
        )
        self.bn2_fix_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1, bitwidth=BITWIDTH,
        )
        self.bn3_fix_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1, bitwidth=BITWIDTH,
        )
        '''
        activation_num = 10 if not downsample else 11
        self.fix_params = [
            _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
            for _ in range(activation_num)
        ]

        # initialize activation fix modules
        for i in range(len(self.fix_params)):
            setattr(self, "fix"+str(i), nnf.Activation_fix(nf_fix_params=self.fix_params[i]))

        '''
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        '''
        self.conv1 = nnf.Conv2d_fix(inplanes, planes, kernel_size=1, bias=False, nf_fix_params=self.conv1_fix_params)
        #self.bn1 = nnf.BatchNorm2d_fix(planes, nf_fix_params=self.bn1_fix_params)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nnf.Conv2d_fix(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, nf_fix_params=self.conv2_fix_params)
        #self.bn2 = nnf.BatchNorm2d_fix(planes, nf_fix_params=self.bn2_fix_params)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nnf.Conv2d_fix(planes, planes * 4, kernel_size=1, bias=False, nf_fix_params=self.conv3_fix_params)
        #self.bn3 = nnf.BatchNorm2d_fix(planes * 4, nf_fix_params=self.bn3_fix_params)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fix0(x)
        residual = x

        out = self.fix1(self.conv1(x))
        out = self.fix2(self.bn1(out))
        out = self.fix3(self.relu(out))

        out = self.fix4(self.conv2(out))
        out = self.fix5(self.bn2(out))
        out = self.fix6(self.relu(out))

        out = self.fix7(self.conv3(out))
        out = self.fix8(self.bn3(out))

        if self.downsample is not None:
            residual = self.fix11(self.downsample(x))

        out += residual
        out = self.fix9(self.relu(out))

        return out


class ResNet(nnf.FixTopModule):
    def __init__(self, depth, num_classes=10, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        # initialize fix configurations
        self.conv1_fix_params = _generate_default_fix_cfg(
            ["weight"], method=1, bitwidth=BITWIDTH)
        self.conv2_fix_params = _generate_default_fix_cfg(
            ["weight"], method=1, bitwidth=BITWIDTH)
        '''
        self.bn1_fix_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1, bitwidth=BITWIDTH,
        )
        self.bn2_fix_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1, bitwidth=BITWIDTH,
        )
        '''
        self.fc_fix_params = _generate_default_fix_cfg(
            ["weight", "bias"], method=1, bitwidth=BITWIDTH)
        self.fix_params = [
            _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
            for _ in range(6)
        ]

        self.inplanes = 16
        self.conv1 = nnf.Conv2d_fix(3, 16, kernel_size=3, padding=1,
                               bias=False, nf_fix_params=self.conv1_fix_params)
        #self.bn1 = nnf.BatchNorm2d_fix(16, nf_fix_params=self.bn1_fix_params)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nnf.Linear_fix(64 * block.expansion, num_classes, nf_fix_params=self.fc_fix_params)

        # initialize activation fix modules
        for i in range(len(self.fix_params)):
            setattr(self, "fix"+str(i), nnf.Activation_fix(nf_fix_params=self.fix_params[i]))

        for m in self.modules():
            if isinstance(m, nnf.Conv2d_fix):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nnf.Conv2d_fix(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, nf_fix_params=self.conv2_fix_params),
                #nnf.BatchNorm2d_fix(planes * block.expansion, nf_fix_params=self.bn2_fix_params),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fix0(x)
        x = self.fix1(self.conv1(x))
        x = self.fix2(self.bn1(x))
        x = self.fix3(self.relu(x))    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.fix4(self.avgpool(x))
        x = x.view(x.size(0), -1)
        x = self.fix5(self.fc(x))

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn.bricks.activation import ACTIVATION_LAYERS, build_activation_layer
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn import ConvModule, constant_init, kaiming_init


@ACTIVATION_LAYERS.register_module()
class Mish(nn.Module):
    """"
    Mish activation
    """
    def __init__(self, inplace=None):
        super().__init__()

    def forward(self, x):
        """
        run forward
        """
        out = x * (torch.tanh(F.softplus(x)))
        return out


class Conv(ConvModule):
    # Standard convolution
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Mish'),
                 **kwargs):
        super(Conv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            groups=groups,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 groups=1,
                 expansion=0.5,
                 **kwargs):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(in_channels,
                          hidden_channels,
                          kernel_size=1,
                          **kwargs)
        self.conv2 = Conv(hidden_channels,
                          out_channels,
                          kernel_size=3,
                          groups=groups,
                          **kwargs)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))


class CSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=1,
                 shortcut=True,
                 groups=1,
                 expansion=0.5,
                 csp_act_cfg=dict(type='Mish'),
                 **kwargs):
        super(CSP, self).__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv4 = Conv(2 * hidden_channels, out_channels, kernel_size=1, **kwargs)
        csp_norm_cfg = kwargs.get('norm_cfg', dict(type='BN')).copy()
        self.bn = build_norm_layer(csp_norm_cfg, 2 * hidden_channels)[-1]
        csp_act_cfg_ = csp_act_cfg.copy()
        if csp_act_cfg_['type'] not in ['Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish']:
            csp_act_cfg_.setdefault('inplace', True)
        self.csp_act = build_activation_layer(csp_act_cfg_)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels,
                         hidden_channels,
                         shortcut,
                         groups,
                         expansion=1.0,
                         **kwargs) for _ in range(repetition)])

    def forward(self, x):
        y1 = self.conv3(self.bottlenecks(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.csp_act(self.bn(torch.cat((y1, y2), dim=1))))


class CSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=1,
                 shortcut=False,
                 groups=1,
                 csp_act_cfg=dict(type='Mish'),
                 **kwargs):
        super(CSP2, self).__init__()
        hidden_channels = int(out_channels)  # hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = Conv(2 * hidden_channels, out_channels, kernel_size=1, **kwargs)

        csp_norm_cfg = kwargs.get('norm_cfg', dict(type='BN')).copy()
        self.bn = build_norm_layer(csp_norm_cfg, 2 * hidden_channels)[-1]
        csp_act_cfg_ = csp_act_cfg.copy()
        if csp_act_cfg_['type'] not in ['Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish']:
            csp_act_cfg_.setdefault('inplace', True)
        self.csp_act = build_activation_layer(csp_act_cfg_)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels,
                         hidden_channels,
                         shortcut,
                         groups,
                         expansion=1.0,
                         **kwargs) for _ in range(repetition)])

    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.bottlenecks(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.csp_act(self.bn(torch.cat((y1, y2), dim=1))))


class CSP3(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=1,
                 shortcut=False,
                 groups=1,
                 csp_act_cfg=dict(type='Mish'),
                 expansion=0.5,
                 **kwargs):
        super(CSP3, self).__init__()
        hidden_channels = int(out_channels*expansion)  # hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, **kwargs)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1, **kwargs)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1, **kwargs)
        
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels,
                         hidden_channels,
                         shortcut,
                         groups,
                         expansion=1.0,
                         **kwargs) for _ in range(repetition)])

    def forward(self, x):
        y1 = self.bottlenecks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))

class Focus(nn.Module):
    # Focus wh information into c-space
    # Implement with ordinary Conv2d with doubled kernel/padding size & stride 2
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 **kwargs):
        super(Focus, self).__init__()
        # padding = kernel_size // 2
        # kernel_size *= 2
        # padding *= 2
        # stride *= 2
        self.conv = Conv(in_channels*4,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                        #  padding=padding,
                         groups=groups,
                         **kwargs)

    def forward(self, x):
        y1 = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        y2 = self.conv(y1)
        return y2
        
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self,
                 in_channels,
                 out_channels,
                 pooling_kernel_size=(5, 9, 13),
                 **kwargs):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = Conv(hidden_channels * (len(pooling_kernel_size) + 1), out_channels, kernel_size=1, **kwargs)
        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in pooling_kernel_size])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [maxpool(x) for maxpool in self.maxpools], 1))


class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 pooling_kernel_size=(5, 9, 13),
                 csp_act_cfg=dict(type='Mish'),
                 **kwargs):
        super(SPPCSP, self).__init__()
        hidden_channels = int(2 * out_channels * expansion)  # hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = Conv(hidden_channels, hidden_channels, kernel_size=3, **kwargs)
        self.conv4 = Conv(hidden_channels, hidden_channels, kernel_size=1, **kwargs)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in pooling_kernel_size])
        self.conv5 = Conv(4 * hidden_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv6 = Conv(hidden_channels, hidden_channels, kernel_size=3, **kwargs)
        csp_norm_cfg = kwargs.get('norm_cfg', dict(type='BN')).copy()
        self.bn = build_norm_layer(csp_norm_cfg, 2 * hidden_channels)[-1]
        csp_act_cfg_ = csp_act_cfg.copy()
        if csp_act_cfg_['type'] not in ['Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish']:
            csp_act_cfg_.setdefault('inplace', True)
        self.csp_act = build_activation_layer(csp_act_cfg_)
        self.conv7 = Conv(2 * hidden_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(self.conv5(torch.cat([x1] + [maxpool(x1) for maxpool in self.maxpools], 1)))
        y2 = self.conv2(x)
        return self.conv7(self.csp_act(self.bn(torch.cat((y1, y2), dim=1))))

# used in yolov4
# 3x conv, 5x conv ...
class ConvList(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=3,
                 **kwargs):
        super(ConvList, self).__init__()
        conv_list = []
        conv_list.append(
            Conv(in_channels, out_channels, kernel_size=1, stride=1, **kwargs)
        )
        for i in range(int((repetition-1)/2)):
            conv_list.extend([
                Conv(out_channels, out_channels*2, kernel_size=3, stride=1, **kwargs),
                Conv(out_channels*2, out_channels, kernel_size=1, stride=1, **kwargs),
            ])
        self.convs = nn.ModuleList(conv_list)

    def forward(self, x):
        y = self.convs(x)
        return y

# used in yolov4
# conv*3 -> spp -> conv*3
class SPPConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=3,
                 pooling_kernel_size=(5, 9, 13),
                 **kwargs):
        super(SPPConv, self).__init__()
        self.conv_3x = ConvList(in_channels, out_channels, repetition, **kwargs)
        self.spp = SPP(out_channels, out_channels*4, pooling_kernel_size, **kwargs)
        self.conv2_3x = ConvList(out_channels*4, out_channels, repetition, **kwargs)
    
    def forward(self, x):
        y1 = self.conv_3x(x)
        y2 = self.spp(y1)
        y3 = self.conv2_3x(y2)
        return y3

# used in yolov5
# conv -> spp -> csp
class SPPCSP3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=1,
                 expansion=0.5,
                 pooling_kernel_size=(5, 9, 13),
                 csp_act_cfg=dict(type='Mish'),
                 **kwargs):
        super(SPPCSP3, self).__init__()
        # downsample
        self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=2, **kwargs)
        # spp
        self.spp = SPP(out_channels, out_channels, pooling_kernel_size, **kwargs)
        # csp
        self.csp = CSP3(out_channels, out_channels, repetition, **kwargs)
    
    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.spp(y1)
        y3 = self.csp(y2)
        return y3


class CSPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition,
                 csp_type='csp',
                 **kwargs):
        super(CSPBlock, self).__init__()
        # downsample
        self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=2, **kwargs)
        # csp
        if csp_type == 'csp':
            self.csp = CSP(out_channels, out_channels, repetition, **kwargs)
        elif csp_type == 'csp2':
            self.csp = CSP2(out_channels, out_channels, repetition, **kwargs)
        elif csp_type == 'csp3':
            self.csp = CSP3(out_channels, out_channels, repetition, **kwargs)
        else:
            print("Not support this csp block:", csp_type)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.csp(y1)
        return y2

class SPPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition,
                 spp_type='spp_conv',
                 **kwargs):
        super(SPPBlock, self).__init__()
        # spp
        if spp_type == 'spp_conv': # used in yolov4
            self.spp = SPPConv(in_channels, out_channels, repetition, **kwargs)
        elif spp_type == 'spp_csp3': # used in yolov5
            self.spp = SPPCSP3(in_channels, out_channels, **kwargs)
        else:
            print("Not support this spp block:", spp_type)

    def forward(self, x):
        y = self.spp(x)
        return y
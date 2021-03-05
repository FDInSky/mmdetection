import logging
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmdet.models.builder import BACKBONES
from .base_module import * 


arch_settings = {
    'yolov4' : {
        's5p': [['conv', 'bottleneck', 'csp', 'csp', 'csp', 'spp_conv'],
                [None, 1, 1, 3, 3, 1, 3],
                [16, 32, 64, 128, 256, 512, 512]],
        'm5p': [['conv', 'bottleneck', 'csp', 'csp', 'csp', 'spp_conv'],
                [None, 1, 1, 5, 5, 3, 3],
                [24, 48, 96, 192, 384, 768, 768]],
        'l5p': [['conv', 'csp', 'csp', 'csp', 'csp', 'csp', 'spp_conv'],
                [None, 1, 2, 8, 8, 4, 3],
                [32, 64, 128, 256, 512, 1024, 1024]],
        'x5p': [['conv', 'bottleneck', 'csp', 'csp', 'csp', 'spp_conv'],
                [None, 1, 3, 11, 11, 5],
                [40, 80, 160, 320, 640, 1280, 1280]],
        '6p': [['conv', 'csp', 'csp', 'csp', 'csp', 'csp', 'csp', 'spp_conv'],
               [None, 1, 3, 15, 15, 7, 7, 3],
               [32, 64, 128, 256, 512, 1024, 1024, 1024]],
        '7p': [['conv', 'csp', 'csp', 'csp', 'csp', 'csp', 'csp', 'csp', 'spp_conv'],
               [None, 1, 3, 15, 15, 7, 7, 7, 3],
               [40, 80, 160, 320, 640, 1280, 1280, 1280, 1280]],
    },
    'yolov5' : {
        's5p': [['focus', 'csp3', 'csp3', 'csp3', 'spp_csp3'],
                [None, 3, 9, 9, 3],
                [64, 128, 256, 512, 1024]],
        'm5p': [['focus', 'csp3', 'csp3', 'csp3', 'spp_csp3'],
                [None, 3, 9, 9, 3],
                [64, 128, 256, 512, 1024]],
        'l5p': [['focus', 'csp3', 'csp3', 'csp3', 'spp_csp3'],
                [None, 3, 9, 9, 3],
                [64, 128, 256, 512, 1024]],
        'x5p': [['focus', 'csp3', 'csp3', 'csp3', 'spp_csp3'],
                [None, 3, 9, 9, 3],
                [64, 128, 256, 512, 1024]],
    }
}

@BACKBONES.register_module()
class DarknetCSP(nn.Module):
    """Darknet backbone.
    Args:
        scale (int): scale of DarknetCSP. 's'|'x'|'m'|'l'|
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Mish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    """

    def __init__(self,
                 architecture='yolov5',
                 scale='x5p',
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True, eps=0.001, momentum=0.03),
                 act_cfg=dict(type='Mish'),
                 csp_act_cfg=dict(type='Mish'),
                 norm_eval=False):
        super(DarknetCSP, self).__init__()
        global arch_settings
        arch_settings = arch_settings[architecture]
        if isinstance(scale, str):
            if scale not in arch_settings:
                raise KeyError(f'invalid scale {scale} for DarknetCSP')
            stage, repetition, channels = arch_settings[scale]
        else:
            stage, repetition, channels = scale

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg, csp_act_cfg=csp_act_cfg)

        self.layers = []
        cin = 3
        for i, (stg, rep, cout) in enumerate(zip(stage, repetition, channels)):
            layer_name = f'{stg}{i}'
            self.layers.append(layer_name)
            if stg == 'conv':
                self.add_module(layer_name, Conv(cin, cout, 3, **cfg))
            elif stg == 'bottleneck':
                self.add_module(layer_name, BottleneckStage(cin, cout, rep, **cfg))
            elif stg == 'focus':
                self.add_module(layer_name, Focus(cin, cout, 3, 1, **cfg))
            elif stg == 'spp':
                self.add_module(layer_name, SPP(cin, cout, **cfg))
            elif stg in ['csp', 'csp2', 'csp3']:
                self.add_module(layer_name, CSPBlock(cin, cout, rep, stg, **cfg))
            elif stg in ['spp_conv', 'spp_csp3']:
                self.add_module(layer_name, SPPBlock(cin, cout, rep, stg, **cfg))
            else:
                raise NotImplementedError
            cin = cout

        self.norm_eval = norm_eval
        self._freeze_stages()
        self.fp16_enabled = False

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            # print("Debug backbone: ", i, layer_name)
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(0, self.frozen_stages):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(DarknetCSP, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
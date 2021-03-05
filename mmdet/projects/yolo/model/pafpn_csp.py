import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmcv.cnn import xavier_init
from mmdet.models.builder import NECKS
from .base_module import Conv, CSP2, CSP3, ConvList


@NECKS.register_module()
class YOLOV4PAFPNCSP(nn.Module):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int or List[int]): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=None,
                 convs_repetition=5,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=dict(type='BN', requires_grad=True, eps=0.001, momentum=0.03),
                 act_cfg=dict(type='Mish'),
                 csp_act_cfg=dict(type='Mish'),
                 upsample_cfg=dict(mode='nearest')):
        super(YOLOV4PAFPNCSP, self).__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        if isinstance(out_channels, list):
            self.out_channels = out_channels
            num_outs = len(out_channels)
        else:
            assert num_outs is not None
            self.out_channels = [out_channels] * num_outs

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs == self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg, csp_act_cfg=csp_act_cfg)

        # top-down path
        # P5 -> P4 -> P3
        self.top_down_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.top_down_csps = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level): # [1, 2]
            bottom_channels = in_channels[i - 1] // 2
            top_channels = in_channels[i] // 2
            pre_conv = Conv(
                in_channels=top_channels,
                out_channels=bottom_channels,
                kernel_size=1,
                **cfg)
            lat_conv = Conv(
                in_channels=2 * bottom_channels,
                out_channels=bottom_channels,
                kernel_size=1,
                **cfg)
            # 5 * conv
            post_conv = ConvList(
                in_channels=2 * bottom_channels,
                out_channels=bottom_channels,
                repetition=convs_repetition,
            )
            self.top_down_convs.append(pre_conv)
            self.lateral_convs.append(lat_conv)
            self.top_down_csps.append(post_conv)

        # bottom-up path
        # P3 -> P4 -> P5
        self.bottom_up_convs = nn.ModuleList()
        self.bottom_up_csps = nn.ModuleList()
        for i in range(self.backbone_end_level - self.start_level - 1): # [2, 1]
            bottom_channels = self.in_channels[self.start_level + i] // 2
            top_channels = self.in_channels[self.start_level + i + 1] // 2
            down_conv = Conv(
                in_channels=bottom_channels,
                out_channels=top_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                **cfg)
            # 5 * conv
            post_conv = ConvList(
                in_channels=2 * top_channels,
                out_channels=top_channels,
                repetition=convs_repetition,
            )
            self.bottom_up_convs.append(down_conv)
            self.bottom_up_csps.append(post_conv)

        # output conv
        self.out_convs = nn.ModuleList()
        for i in range(num_outs):
            before_conv_channels = self.in_channels[self.start_level + i] // 2
            out_channels = self.out_channels[i]
            out_conv = Conv(
                in_channels=before_conv_channels,
                out_channels=out_channels,
                kernel_size=3,
                **cfg
            )
            self.out_convs.append(out_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        used_backbone_levels = self.backbone_end_level - self.start_level

        # build top-down path
        x = inputs[self.backbone_end_level - 1]
        bottom_up_merge = []
        for i in range(used_backbone_levels - 1, 0, -1):
            pre_conv = self.top_down_convs[i - 1]
            lat_conv = self.lateral_convs[i - 1]
            post_conv = self.top_down_csps[i - 1]
            bottom_up_merge.append(x)
            inputs_bottom = lat_conv(inputs[self.start_level + i - 1])
            # conv
            x = pre_conv(x)
            # upsample
            if 'scale_factor' in self.upsample_cfg:
                x = F.interpolate(x, **self.upsample_cfg)
            else:
                bottom_shape = inputs_bottom.shape[2:]
                x = F.interpolate(x, size=bottom_shape, **self.upsample_cfg)
            # concat
            x = torch.cat((inputs_bottom, x), dim=1)
            # conv
            x = post_conv(x)

        # build bottom-up path
        outs = [x]
        for i in range(self.backbone_end_level - self.start_level - 1):
            down_conv = self.bottom_up_convs[i]
            post_conv = self.bottom_up_csps[i]
            x = down_conv(x)
            x = torch.cat((x, bottom_up_merge.pop(-1)), dim=1)
            x = post_conv(x)
            outs.append(x)

        # build output
        for i in range(len(outs)):
            outs[i] = self.out_convs[i](outs[i])

        return tuple(outs)


@NECKS.register_module()
class PAFPNCSP(nn.Module):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int or List[int]): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=None,
                 csp_repetition=3,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=dict(type='BN', requires_grad=True, eps=0.001, momentum=0.03),
                 act_cfg=dict(type='Mish'),
                 csp_act_cfg=dict(type='Mish'),
                 upsample_cfg=dict(mode='nearest')):

        super(PAFPNCSP, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        if isinstance(out_channels, list):
            self.out_channels = out_channels
            num_outs = len(out_channels)
        else:
            assert num_outs is not None
            self.out_channels = [out_channels] * num_outs

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs == self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg, csp_act_cfg=csp_act_cfg)

        # top-down path
        # P5 -> P4 -> P3
        self.top_down_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.top_down_csps = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):  #[1, 2]
            top_channels = in_channels[i]
            bottom_channels = in_channels[i] // 2
            # print("debug top-down: ", i, top_channels, bottom_channels)
            pre_conv = Conv(
                in_channels=top_channels,
                out_channels=bottom_channels,
                kernel_size=1,
                **cfg)
            up_conv = Conv(
                in_channels=bottom_channels,
                out_channels=bottom_channels,
                kernel_size=1,
                **cfg)
            post_csp = CSP3(
                in_channels=2 * bottom_channels,
                out_channels=bottom_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg
            )
            self.top_down_convs.append(pre_conv)
            self.lateral_convs.append(up_conv)
            self.top_down_csps.append(post_csp)

        # bottom-up path
        # P3 -> P4 -> P5
        self.bottom_up_convs = nn.ModuleList()
        self.bottom_up_csps = nn.ModuleList()
        for i in range(self.backbone_end_level - self.start_level - 1):
            bottom_channels = self.in_channels[self.start_level + i]
            top_channels = self.in_channels[self.start_level + i + 1]
            # print("Debug bottom-up: ", i, bottom_channels, top_channels)
            down_conv = Conv(
                in_channels=bottom_channels,
                out_channels=top_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                **cfg)
            post_csp = CSP3(
                in_channels=2 * top_channels,
                out_channels=top_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg)
            self.bottom_up_convs.append(down_conv)
            self.bottom_up_csps.append(post_csp)

        # output conv
        self.out_convs = nn.ModuleList()
        for i in range(num_outs):
            before_conv_channels = self.in_channels[self.start_level + i]
            out_channels = self.out_channels[i]
            out_conv = Conv(
                in_channels=before_conv_channels,
                out_channels=out_channels,
                kernel_size=3,
                **cfg
            )
            self.out_convs.append(out_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        used_backbone_levels = self.backbone_end_level - self.start_level

        # build top-down path
        x = inputs[self.backbone_end_level - 1]
        bottom_up_merge = []
        for i in range(used_backbone_levels - 1, 0, -1):  # [2, 1]
            pre_conv = self.top_down_convs[i - 1]
            up_conv = self.lateral_convs[i - 1]
            post_csp = self.top_down_csps[i - 1]
            bottom_up_merge.append(x)
            inputs_bottom = up_conv(inputs[self.start_level + i - 1])
            # conv
            x = pre_conv(x)
            # upsample
            if 'scale_factor' in self.upsample_cfg:
                x = F.interpolate(x, **self.upsample_cfg)
            else:
                bottom_shape = inputs_bottom.shape[2:]
                x = F.interpolate(x, size=bottom_shape, **self.upsample_cfg)
            # concat
            x = torch.cat((inputs_bottom, x), dim=1)
            # csp
            x = post_csp(x)

        # build bottom-up path
        outs = [x]
        for i in range(self.backbone_end_level - self.start_level - 1):
            down_conv = self.bottom_up_convs[i]
            post_csp = self.bottom_up_csps[i]
            x = down_conv(x)
            x = torch.cat((x, bottom_up_merge.pop(-1)), dim=1)
            x = post_csp(x)
            outs.append(x)

        # build output
        for i in range(len(outs)):
            outs[i] = self.out_convs[i](outs[i])

        return tuple(outs)

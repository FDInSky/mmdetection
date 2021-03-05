import torch 
import torch.nn as nn 
from mmcv.cnn import normal_init, ConvModule
from mmcv.ops import DeformConv2d 
    

class ROIDCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 deform_groups=1,
                 kernel_size=3,
                 offset_type='bbox2offset'
                 ):
        super(ROIDCN, self).__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.deform_conv = DeformConv2d(in_channels,
                                        out_channels,
                                        kernel_size=self.kernel_size,
                                        padding=self.pad,
                                        groups=deform_groups,
                                        deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)
        self.offset_type = offset_type
        if offset_type == 'feat2offset':
            self.conv_offset = nn.Conv2d(in_channels, self.kernel_size ** 2 * 2, 1, bias=False)

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    def forward(self, x, bboxes, stride, scores):
        if self.offset_type == 'bbox2offset':
            offset = self.bbox2offset(bboxes, self.kernel_size, stride, scores)
            # print("debug1: ", x.shape, offset.shape)
            # torch.Size([4, 256, 2, 11]) torch.Size([4, 18, 2, 11])
        elif self.offset_type == 'feat2offset':
            offset = self.conv_offset(x)

        x = self.relu(self.deform_conv(x, offset))
        return x

    def bbox2offset(self, bboxes, kernel_size, stride, scores):
        num_imgs, H, W, _ = bboxes.shape
        featmap_size = (H, W)
        offset_list = []
        for i in range(num_imgs):
            bbox = bboxes[i].reshape(-1, 4)  #(NA, 4)
            score = scores[i].reshape(-1, 1)
            offset = self.bbox2offset_single(bbox, kernel_size, featmap_size, stride, score)
            # print("debug offset: ", offset.shape)
            # debug offset:  torch.Size([4704, 9, 2])
            offset = offset.reshape(bbox.size(0), -1).permute(1, 0).reshape(-1, H, W) # [2*ks**2,H,W]
            offset_list.append(offset)
        offset_tensor = torch.stack(offset_list, dim=0)
        return offset_tensor

    def bbox2offset_single(self, bbox, kernel_size, featmap_size, stride, score):
        dtype, device = bbox.dtype, bbox.device
        feat_h, feat_w = featmap_size

        # get padding of every grid location
        idx = torch.arange(-self.pad, self.pad + 1, dtype=dtype, device=device)
        # idx = torch.arange(-self.pad, self.pad, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy
        # print("debug: ", xx.shape, yy.shape, xc.shape, yc.shape, x_conv.shape, y_conv.shape)
        # debug:  torch.Size([9]) torch.Size([9]) torch.Size([4704]) torch.Size([4704]) torch.Size([4704, 9]) torch.Size([4704, 9])
        
        # get sampling locations of bboxes
        # TODO: optim 1rst stage box encode method
        # x_ctr, y_ctr, w, h = torch.unbind(bboxes, dim=1)
        x1, y1, x2, y2 = torch.unbind(bbox, dim=1)
        x_ctr = (x1 + x2) * 0.5
        y_ctr = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y2
        # box in every feature location 
        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        # add padding*scale
        dw, dh = w_s / kernel_size, h_s / kernel_size
        x, y = dw[:, None]*xx, dh[:, None]*yy
        x_bbox, y_bbox = x + x_ctr[:, None], y + y_ctr[:, None]

        # get offset filed between box_center and conv_center
        offset_x = x_bbox - x_conv
        offset_y = y_bbox - y_conv
        # print("offset xy: ", offset_x.shape, offset_y.shape)

        # do weight
        offset_x = offset_x * score
        offset_y = offset_y * score
        # x, y in bboxes is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        
        return offset

class RefineFeat(nn.Module):
    def __init__(
            self,
            in_channels,
            featmap_strides,
            deform_groups=1,
            conv_cfg=None,
            norm_cfg=None):
        super(RefineFeat, self).__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.deform_groups = deform_groups
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        
    def _init_layers(self):
        self.mlvl_roi_dcn = nn.ModuleList(
            [ROIDCN(self.in_channels, self.in_channels, self.deform_groups) for s in self.featmap_strides]
        )

    def init_weights(self):
        pass 

    def forward(self, x, rois, rois_score):
        mlvl_rois = [torch.cat(best_rbbox) for best_rbbox in zip(*rois)]
        mlvl_scores = [torch.cat(best_rbbox_score) for best_rbbox_score in zip(*rois_score)]

        outs = []
        for roi_dcn, x_scale, roi, score, f_stride in zip(self.mlvl_roi_dcn, x, mlvl_rois, mlvl_scores, self.featmap_strides):
            N, C, H, W = x_scale.shape
            # print("debug feat adapt: ", x_scale.shape, roi.shape, score.shape)
            roi = roi.view(N, H, W, 4)
            # align feat
            refined_feat = roi_dcn(x_scale, roi, f_stride, score)
            outs.append(refined_feat)

        return outs


    
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d 
from mmdet.core import (anchor_inside_flags, multiclass_nms, unmap, multi_apply, images_to_levels)
from mmdet.models.dense_heads import AnchorHead
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module
class FCRHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)
                 ),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=1.0, # hard & easy
                     alpha=0.5, # pos & neg
                     loss_weight=1.0),
                 loss_iou=dict(
                     type='CIoULoss',
                     loss_weight=1.0),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        super(FCRHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn, self.feat_channels, 3, stride=1, padding=1,
                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
                )
            self.reg_convs.append(
                ConvModule(
                    chn, self.feat_channels, 3, stride=1, padding=1,
                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
                )
        # decode feat into output
        self.box_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.box_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.box_cls, std=0.01, bias=bias_cls)
        normal_init(self.box_reg, std=0.01)

    def forward_single(self, x):
        # head feat
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        # decode head feat
        cls_score = self.box_cls(cls_feat)
        reg_delta = self.box_reg(reg_feat)
        return cls_score, reg_delta

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)

        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            num_level_anchors=None,
                            label_channels=1,
                            unmap_outputs=True):
        
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        
        if not inside_flags.any():
            return (None,) * 6
        
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)

        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_targets_iou = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # print("debug pos&neg: ", pos_inds.shape, neg_inds.shape)
        total_inds = len(pos_inds) + len(neg_inds)
        if total_inds == 0:
            relative_num_diff = 0.0
        else:
            relative_num_diff = abs(len(neg_inds) - len(pos_inds)) / total_inds
        if len(pos_inds) > 0:
            bbox_targets_iou[pos_inds, :] = sampling_result.pos_gt_bboxes.float()
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

            if self.train_cfg.pos_weight <= 0:
                # label_weights[pos_inds] = 1.0 
                label_weights[pos_inds] = 1 + relative_num_diff
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            # label_weights[neg_inds] = 1.0
            label_weights[neg_inds] = 1 - relative_num_diff

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            bbox_targets_iou = unmap(bbox_targets_iou, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result, bbox_targets_iou)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            num_level_anchors_list,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list, all_bbox_targets_iou) = results[:8]
        rest_results = list(results[8:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_targets_iou_list = images_to_levels(all_bbox_targets_iou, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg, bbox_targets_iou_list)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)
        
        return res + tuple(rest_results)
    
    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, bbox_targets_iou_list) = cls_reg_targets[:7]
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_targets_iou_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_targets_iou, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1).contiguous()
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        # print("debug3: ", cls_score.shape, labels.shape, label_weights.shape)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4).contiguous()
        bbox_weights = bbox_weights.reshape(-1, 4).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)
        # iou loss
        bbox_targets_iou = bbox_targets_iou.reshape(-1, 4).contiguous()
        anchors = anchors.reshape(-1, 4)
        bbox_pred_iou = self.bbox_coder.decode(anchors, bbox_pred)
        loss_iou = self.loss_iou(bbox_pred_iou, bbox_targets_iou, bbox_weights)

        # return loss_cls, loss_bbox+loss_iou
        return loss_cls, loss_iou

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        
        cfg = self.test_cfg['s0'] if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self,
                      cls_scores,
                      bbox_preds):
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        bboxes_list = [[] for _ in range(num_imgs)]
        scores_list = [[] for _ in range(num_imgs)]
        

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]
            
            N, _, H, W = cls_score.shape
            cls_score = cls_score.permute(0, 2, 3, 1)  # (N, H, W, A*C)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors, self.cls_out_channels)  # (N, H*W, A, C)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            scores, _ = scores.max(dim=-1, keepdim=True)  # (N, H*W, A, 1)
            best_ind = scores.argmax(dim=-2, keepdim=True)  # (N, H*W, 1, 1)
            # select best bbox pred
            best_score = cls_score.gather(dim=-2, index=best_ind).squeeze(dim=-2) # (N, H*W, 1)
            best_bbox_ind = best_ind.expand(-1, -1, -1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # (N, H, W, A*4)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 4)  # (N, H*W, A, 4)
            best_pred = bbox_pred.gather(dim=-2, index=best_bbox_ind).squeeze(dim=-2)  # (N, H*W, 4)
            # print("debug best: ", best_pred.shape, best_score.shape)
            # select best anchor
            anchors = mlvl_anchors[lvl] # (H*W*A, 4)
            anchors = anchors.reshape(-1, self.num_anchors, 4)  # (H*W, A, 4)

            for img_id in range(num_imgs):
                best_pred_i = best_pred[img_id] # (H*W, 4)
                best_score_i = best_score[img_id] # (H*W, 1)
                best_ind_i = best_bbox_ind[img_id] # (H*W, 1, 4)
                best_anchor_i = anchors.gather(dim=-2, index=best_ind_i).squeeze(dim=-2)  # (H*W, 4)
                # print("debug filter: ", best_anchor_i.shape, best_pred_i.shape, best_score_i.shape)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i, best_pred_i)

                bboxes_list[img_id].append(best_bbox_i.detach())
                scores_list[img_id].append(best_score_i.detach())
        
        return bboxes_list, scores_list


@HEADS.register_module
class FCRBinaryHead(FCRHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)
                 ),
                 loss_iou=dict(type='IoULoss', loss_weight=1.0),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        super(FCRBinaryHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            **kwargs)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            num_level_anchors=None,
                            label_channels=1,
                            unmap_outputs=True):
        
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        
        if not inside_flags.any():
            return (None,) * 6
        
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)

        # binary class 
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, None)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_targets_iou = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        total_inds = len(pos_inds) + len(neg_inds)
        # print("debug pos&neg: ", pos_inds.shape, neg_inds.shape)
        if total_inds == 0:
            relative_num_diff = 0.0
        else:
            relative_num_diff = abs(len(neg_inds) - len(pos_inds)) / total_inds
        if len(pos_inds) > 0:
            bbox_targets_iou[pos_inds, :] = sampling_result.pos_gt_bboxes.float()
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            # binary class
            labels[pos_inds] = 1
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1 + relative_num_diff
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1 - relative_num_diff

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            bbox_targets_iou = unmap(bbox_targets_iou, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result, bbox_targets_iou)


@HEADS.register_module
class FCRRefineHead(FCRHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='PseudoAnchorGenerator',
                     strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 **kwargs):

        self.rois = None
        super(FCRRefineHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            **kwargs)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        anchor_list = [
            [bboxes_img_lvl.clone().detach() for bboxes_img_lvl in bboxes_img]
            for bboxes_img in self.rois]
        # print(len(self.rois), len(self.rois[0]), self.rois[0][0].shape)
        # 8 5 torch.Size([4704, 4])
        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device)
            # print("debug valid flag: ", len(multi_level_flags), len(multi_level_flags[0]))
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def get_anchors_v2(self, featmap_sizes, img_metas, device='cuda'):
        anchor_list = [
            [bboxes_img_lvl.clone().detach() for bboxes_img_lvl in bboxes_img]
            for bboxes_img in self.rois]
        # print(len(self.rois), len(self.rois[0]), self.rois[0][0].shape)
        # 8 5 torch.Size([4704, 4])
        # for each image, we compute valid flags of multi level anchors
        good_flag_list = [[] for bboxes_img in self.rois]
        for img_id, img_meta in enumerate(img_metas):
            for i in range(len(self.rois[0])):
                rois_score = self.rois_scores[img_id][i]
                pos_thres = rois_score.mean()
                rois_mask = rois_score >= pos_thres
                # print("debug loc mask: ", loc_mask.shape)
                rois_mask = rois_mask.contiguous().view(-1)
                good_flag_list[img_id].append(rois_mask)
        return anchor_list, good_flag_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             rois=None,
             rois_scores=None,
             gt_bboxes_ignore=None):

        self.rois = rois
        self.rois_scores = rois_scores
        return super(FCRRefineHead, self).loss(cls_scores=cls_scores,
                                            bbox_preds=bbox_preds,
                                            gt_bboxes=gt_bboxes,
                                            gt_labels=gt_labels,
                                            img_metas=img_metas,
                                            gt_bboxes_ignore=gt_bboxes_ignore)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self,
                      cls_scores,
                      bbox_preds,
                      rois):
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        bboxes_list = [[] for _ in range(num_imgs)]  # list(indexed by images) of list(indexed by levels)
        scores_list = [[] for _ in range(num_imgs)]
        assert rois is not None
        # rois is a list of lists. outer list is indexed by images, while inner lists are indexed by levels
        mlvl_rois = [torch.cat(r) for r in zip(*rois)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            rois = mlvl_rois[lvl]  # (N*H*W, 4)
            assert bbox_pred.size(1) == 4
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # (N, H, W, 4)
            bbox_pred = bbox_pred.reshape(-1, 4)  # (N*H*W, 4)
            refined_bbox = self.bbox_coder.decode(rois, bbox_pred)  # (N*H*W, 4)
            refined_bbox = refined_bbox.reshape(num_imgs, -1, 4)  # (N, H*W, 4)
            
            cls_score = cls_scores[lvl]
            cls_score = cls_score.permute(0, 2, 3, 1)  # (N, H, W, A*C)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors, self.cls_out_channels)  # (N, H*W, A, C)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            scores, _ = scores.max(dim=-1, keepdim=True)  # (N, H*W, A, 1)
            best_ind = scores.argmax(dim=-2, keepdim=True)  # (N, H*W, 1, 1)
            # select best bbox pred
            best_score = cls_score.gather(dim=-2, index=best_ind).squeeze(dim=-2) # (N, H*W, 1)

            for img_id in range(num_imgs):
                bboxes_list[img_id].append(refined_bbox[img_id].detach())
                scores_list[img_id].append(best_score[img_id].detach())
        return bboxes_list, scores_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   rois=None,
                   rois_scores=None,
                   ):
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        result_list = []

        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                rois[img_id], img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list
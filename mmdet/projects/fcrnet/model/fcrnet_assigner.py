import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class FCRAssignerV2(BaseAssigner):
    def __init__(self,
                 topk=100,
                 ratio=0.01,
                 stage=1,
                 iou_type='iou',
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 ):
        self.topk = topk
        self.ratio = ratio
        self.stage = stage
        self.iou_type = 'iou'
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore,
               gt_labels):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes, mode=self.iou_type)
        # init assign gt index as -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ), 0, dtype=torch.long)
    
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # select positive bbox
        # gt_x1 < box_cx < gt_x2, gt_y1 < box_cy < gt_y2
        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        l_ = ep_bboxes_cx.view(-1, num_gt) > gt_bboxes[:, 0]
        t_ = ep_bboxes_cy.view(-1, num_gt) > gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] > ep_bboxes_cx.view(-1, num_gt)
        b_ = gt_bboxes[:, 3] > ep_bboxes_cy.view(-1, num_gt)
        is_in_gt = l_ * t_ * r_ * b_
        # iou greather than mean+std
        index = []
        neg_index = []
        for i in range(num_gt):
            is_in_gt_i = is_in_gt[:, i]
            candidate_idxs_i = torch.where(is_in_gt_i)[0]
            candidate_overlaps_i = overlaps[candidate_idxs_i, i]
            iou_mean = candidate_overlaps_i.mean(0)
            iou_std = candidate_overlaps_i.std(0)
            if self.stage == 0:
                pos_thres = iou_mean
                neg_thres = iou_mean
            else:
                pos_thres = iou_mean+iou_std
                neg_thres = iou_mean+iou_std
            
            # print("debug iou thresh: ", self.stage, iou_mean, iou_std)
            # positive 
            is_pos_i = candidate_overlaps_i >= pos_thres
            index_i = candidate_idxs_i[is_pos_i]
            index.append(index_i)

        index = torch.cat(index)

        # assign gt index
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        # positive gt
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        # assign labels
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), 0)
        # positive lables
        if self.stage > 0:
            # multi-class label
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            # binary-class label positive
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = 1

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

@BBOX_ASSIGNERS.register_module()
class FCRAssigner(BaseAssigner):
    def __init__(self,
                 topk=100,
                 ratio=0.01,
                 stage=1,
                 iou_type='iou',
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 ):
        self.topk = topk
        self.ratio = ratio
        self.stage = stage
        self.iou_type = 'iou'
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore,
               gt_labels):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes, mode=self.iou_type)
        # init assign gt index as -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ), 0, dtype=torch.long)
    
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # select positive bbox
        # gt_x1 < box_cx < gt_x2, gt_y1 < box_cy < gt_y2
        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        l_ = ep_bboxes_cx.view(-1, num_gt) > gt_bboxes[:, 0]
        t_ = ep_bboxes_cy.view(-1, num_gt) > gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] > ep_bboxes_cx.view(-1, num_gt)
        b_ = gt_bboxes[:, 3] > ep_bboxes_cy.view(-1, num_gt)
        is_in_gt = l_ * t_ * r_ * b_
        # iou greather than mean+std
        index = []
        neg_index = []
        for i in range(num_gt):
            is_in_gt_i = is_in_gt[:, i]
            candidate_idxs_i = torch.where(is_in_gt_i)[0]
            candidate_overlaps_i = overlaps[candidate_idxs_i, i]
            iou_mean = candidate_overlaps_i.mean(0)
            iou_std = candidate_overlaps_i.std(0)
            if self.stage == 0:
                pos_thres = iou_mean#+iou_std
                neg_thres = iou_mean#+iou_std
            else:
                pos_thres = iou_mean
                neg_thres = iou_mean
            
            # print("debug iou thresh: ", self.stage, iou_mean, iou_std)
            # positive 
            is_pos_i = candidate_overlaps_i >= pos_thres
            index_i = candidate_idxs_i[is_pos_i]
            index.append(index_i)

        index = torch.cat(index)

        # assign gt index
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        # positive gt
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        # assign labels
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), 0)
        # positive lables
        if self.stage > 0:
            # multi-class label
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            # binary-class label positive
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = 1

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign_v2(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore,
               gt_labels):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        
        # init assign gt index as -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
    
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # select positive bbox
        # gt_x1 < box_cx < gt_x2, gt_y1 < box_cy < gt_y2
        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        l_ = ep_bboxes_cx.view(-1, num_gt) > gt_bboxes[:, 0]
        t_ = ep_bboxes_cy.view(-1, num_gt) > gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] > ep_bboxes_cx.view(-1, num_gt)
        b_ = gt_bboxes[:, 3] > ep_bboxes_cy.view(-1, num_gt)
        is_in_gt = l_ * t_ * r_ * b_
        # iou greather than mean+std
        index = []
        neg_index = []
        for i in range(num_gt):
            is_in_gt_i = is_in_gt[:, i]
            candidate_idxs_i = torch.where(is_in_gt_i)[0]
            candidate_overlaps_i = overlaps[candidate_idxs_i, i]
            iou_mean = candidate_overlaps_i.mean(0)
            iou_std = candidate_overlaps_i.std(0)
            if self.stage == 0:
                pos_thres = iou_mean
                neg_thres = iou_mean
            else:
                pos_thres = iou_mean+iou_std
                neg_thres = iou_mean+iou_std
            # print("debug iou thresh: ", self.stage, iou_mean, iou_std)

            # positive 
            is_pos_i = candidate_overlaps_i >= pos_thres
            index_i = candidate_idxs_i[is_pos_i]
            index.append(index_i)
            # negative
            is_neg_i = candidate_overlaps_i < neg_thres
            neg_index_i = candidate_idxs_i[is_neg_i]
            neg_index.append(neg_index_i)

        index = torch.cat(index)
        neg_index = torch.cat(neg_index)

        # assign gt index
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        # positive gt
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        # negative gt
        assigned_gt_inds[neg_index] = 0
        
        # assign labels
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        # positive lables
        if self.stage > 0:
            # multi-class label
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            # binary-class label positive
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = 1
        # negative lables
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze()
        if neg_inds.numel() > 0:
            assigned_labels[neg_inds] = 0

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign_v3(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        
        # init assign gt index as -1 by default
        if self.stage == 0:
            assigned_gt_inds = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
        else:
            assigned_gt_inds = overlaps.new_full((num_bboxes, ), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # select positive bbox
        # gt_x1 < box_cx < gt_x2, gt_y1 < box_cy < gt_y2
        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        l_ = ep_bboxes_cx.view(-1, num_gt) > gt_bboxes[:, 0]
        t_ = ep_bboxes_cy.view(-1, num_gt) > gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] > ep_bboxes_cx.view(-1, num_gt)
        b_ = gt_bboxes[:, 3] > ep_bboxes_cy.view(-1, num_gt)
        is_in_gt = l_ * t_ * r_ * b_
        # iou greather than mean+std
        index = []
        neg_index = []
        for i in range(num_gt):
            is_in_gt_i = is_in_gt[:, i]
            candidate_idxs_i = torch.where(is_in_gt_i)[0]
            candidate_overlaps_i = overlaps[candidate_idxs_i, i]
            iou_mean = candidate_overlaps_i.mean(0)
            iou_std = candidate_overlaps_i.std(0)
            if self.stage == 0:
                pos_thres = iou_mean
                neg_thres = iou_mean
            else:
                pos_thres = iou_mean
                neg_thres = iou_mean
            # print("debug iou thresh: ", self.stage, iou_mean, iou_std)
            # positive 
            is_pos_i = candidate_overlaps_i >= pos_thres
            index_i = candidate_idxs_i[is_pos_i]
            index.append(index_i)
            # negative
            is_neg_i = candidate_overlaps_i < neg_thres
            neg_index_i = candidate_idxs_i[is_neg_i]
            neg_index.append(neg_index_i)

        index = torch.cat(index)
        neg_index = torch.cat(neg_index)

        # assign gt index
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        if self.stage == 0:
            assigned_gt_inds[neg_index] = 0
        
        # assign labels
        if self.stage == 0:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        else:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), 0)
        if self.stage > 0:
            # multi-class label
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            # binary-class label positive
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = 1
        if self.stage == 0:
            neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze()
            if neg_inds.numel() > 0:
                assigned_labels[neg_inds] = 0

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

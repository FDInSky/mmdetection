import os.path as osp
import random
import math
import numpy as np
import cv2
from torch.cuda.amp import GradScaler, autocast
import mmcv
from mmcv.runner.dist_utils import master_only
from mmcv.runner import Hook, Fp16OptimizerHook, HOOKS, OptimizerHook
from mmcv.parallel import is_module_wrapper
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import DETECTORS
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.compose import Compose


@DETECTORS.register_module()
class YOLOV4(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_amp=True):
        super(YOLOV4, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
        self.use_amp = use_amp

    def forward_train(self, *wargs, **kwargs):
        if self.use_amp:
            with autocast():
                return super(YOLOV4, self).forward_train(*wargs, **kwargs)
        else:
            return super(YOLOV4, self).forward_train(*wargs, **kwargs)

    def simple_test(self, *wargs, **kwargs):
        if self.use_amp:
            with autocast():
                return super(YOLOV4, self).simple_test(*wargs, **kwargs)
        else:
            return super(YOLOV4, self).simple_test(*wargs, **kwargs)

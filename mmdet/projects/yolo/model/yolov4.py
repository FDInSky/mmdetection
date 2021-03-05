from collections import OrderedDict
import torch
from torch.cuda.amp import autocast
import torch.distributed as dist
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import DETECTORS


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
        super(YOLOV4, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
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

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                # start
                # TODO: find a better method
                loss_value = loss_value.cuda().clone()
                # end
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

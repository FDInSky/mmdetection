from torch.nn.modules.utils import _pair
from mmdet.core.anchor import AnchorGenerator, ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class PseudoAnchorGenerator(AnchorGenerator):
    """Non-Standard pseudo anchor generator that is used to generate valid flags only!
       Calling its grid_anchors() method will raise NotImplementedError!
    """

    def __init__(self,
                 strides):
        self.strides = [_pair(stride) for stride in strides]

    @property
    def num_base_anchors(self):
        return [1 for _ in self.strides]

    def single_level_grid_anchors(self, featmap_sizes, device='cuda'):
        raise NotImplementedError

    def __repr__(self):
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides})'
        return repr_str
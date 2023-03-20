import torch
import torch.nn as nn
from torch.autograd import Function

from mmdet.models.builder import DETECTORS
from mmdet.models import TwoStageDetector

class GradientDecoupleLayer(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None


class AffineLayer(nn.Module):
    def __init__(self, num_channels, bias=False):
        super(AffineLayer, self).__init__()
        weight = torch.FloatTensor(1, num_channels, 1, 1).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)

        self.bias = None
        if bias:
            bias = torch.FloatTensor(1, num_channels, 1, 1).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)

    def forward(self, X):
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(X)
        return out


def decouple_layer(x, _lambda):
    return GradientDecoupleLayer.apply(x, _lambda)

@DETECTORS.register_module()
class DeFASTERRCNN(TwoStageDetector):
    """Implementation of `Decoupled Faster R-CNN_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 base_training = False,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        rpn_backward_scale = 0
        if base_training:    
            print('base training')        
            roi_backward_scale = 0.75
        else:
            roi_backward_scale = 0.01
        self.rpn_backward_scale = rpn_backward_scale
        self.roi_backward_scale = roi_backward_scale
        super(DeFASTERRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_train(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        proposals=None,
                        **kwargs):
            """
            Args:
                img (Tensor): of shape (N, C, H, W) encoding input images.
                    Typically these should be mean centered and std scaled.

                img_metas (list[dict]): list of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    `mmdet/datasets/pipelines/formatting.py:Collect`.

                gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                    shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

                gt_labels (list[Tensor]): class indices corresponding to each box

                gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                    boxes can be ignored when computing the loss.

                gt_masks (None | Tensor) : true segmentation masks for each box
                    used if the architecture supports a segmentation task.

                proposals : override rpn proposals with custom proposals. Use when
                    `with_rpn` is False.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """

            # x are the features from the backbone
            features = self.extract_feat(img)

            losses = dict()

            # RPN forward and loss
            if self.with_rpn:
                decouped_features = decouple_layer(features, self.rpn_backward_scale)
                
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    decouped_features,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_decouped_features = decouple_layer(features, self.roi_backward_scale)
            roi_losses = self.roi_head.forward_train(roi_decouped_features, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            losses.update(roi_losses)

            return losses

   
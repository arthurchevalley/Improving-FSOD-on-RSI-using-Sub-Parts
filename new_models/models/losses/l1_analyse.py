# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0, w=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0
        
    assert pred.size() == target.size()
    if torch.isnan(pred).any():
        
        print('nan pred')
        print(f'beta {beta} w {w}')
        assert not torch.isnan(pred).any()

    if torch.isnan(target).any():
        print('nan trg')
        target[torch.isnan(target)] = 0.0

    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss_a(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    #loss = torch.abs(pred - target)
    if torch.isnan(pred).any():
        print('l1 nan pred')
    if torch.isnan(target).any():
        print('nan trg')
        target[torch.isnan(target)] = 0.0

    loss = torch.abs(pred - target)
    if torch.isnan(loss).any():
        print('nan loss')

    #print(loss)

    return loss


@LOSSES.register_module()
class SmoothL1Loss_analyse(nn.Module):
    """Smooth L1 loss handling NaN prediction

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss_analyse, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            w=self.loss_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        return loss_bbox


@LOSSES.register_module()
class L1Loss_analyse(nn.Module):
    """L1 loss handling NaN predicitions.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss_analyse, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.iter = 0

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        self.iter += 1
        #print(self.iter)
        loss_bbox = self.loss_weight * l1_loss_a(

            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        #experiment.log_metric(key="loss bbox", value=loss_bbox, step=epoch)

        return loss_bbox
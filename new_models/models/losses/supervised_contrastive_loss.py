from typing import Optional

import torch
import torch.nn as nn
from mmdet.models import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from torch import Tensor
from typing_extensions import Literal


def cross_entropy(pred,
                  label,
                  nbr_cls,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses

    lbl_msk = (label == nbr_cls).float() 
    #bg_msk = 2*(label == nbr_cls).float() 
    
    label_argmax = torch.argmax(pred,dim=1)
    
    label_argmax_neg = (label_argmax == nbr_cls).float()
    label_argmax_pos = (label_argmax != nbr_cls).float()
    agn_lbl = torch.stack((label_argmax_pos, label_argmax_neg), dim=1)
    
    # have a 
    #agn_lbl = (lbl_msk + bg_msk).to(int)
    lbl_msk = lbl_msk.long()
    agn_lbl = agn_lbl.float()
    
    #print(dict().shape)
    loss = F.cross_entropy(
        agn_lbl,
        lbl_msk,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class Object_CrossEntropyLoss(nn.Module):

    def __init__(self,
                 nbr_classes,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(Object_CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.nbr_classes = nbr_classes
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.nbr_classes,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class SupervisedContrastiveLoss(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                features: Tensor,
                labels: Tensor,
                ious: Tensor,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = self.loss_weight
        if decay_rate is not None:
            loss_weight = self.loss_weight * decay_rate

        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask with shape [N, N], mask_{i, j}=1
        # if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().to(features.device)

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask *
                              label_mask).sum(1) / label_mask.sum(1)

        keep = ious >= self.iou_threshold
        if keep.sum() == 0:
            # return zero loss
            return per_label_log_prob.sum() * 0
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        coefficient = self.reweight_func(ious)
        coefficient = coefficient[keep]
        if weight is not None:
            weight = weight[keep]
        loss = loss * coefficient
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss_weight * loss

    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay

@LOSSES.register_module()
class DualSupervisedContrastiveLoss(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = self.loss_weight
        if decay_rate is not None:
            loss_weight = self.loss_weight * decay_rate
        if classes_equi is not None:
            classes_equi[nbr_classes]=nbr_classes
            if (labels_1 > nbr_classes).any():
                labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
            if (labels_2 > nbr_classes).any():
                labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

        if len(labels_1.shape) == 1:
            labels_1 = labels_1.reshape(-1, 1)
        if len(labels_2.shape) == 1:
            labels_2 = labels_2.reshape(-1, 1)
        # mask with shape [N, N], mask_{i, j}=1
        # if sample i and sample j have the same label
        label_mask = torch.eq(labels_1, labels_2.T).float().to(features_1.device)

        similarity = torch.div(
            torch.matmul(features_1, features_2.T), self.temperature)

        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()


        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        lbl_length = label_mask.sum(1)
        lbl_length[lbl_length == 0] = 1
        per_label_log_prob = (log_prob *label_mask).sum(1) / lbl_length

        per_label_log_prob = torch.nan_to_num(per_label_log_prob, nan=0.0)

        keep_1 = ious_1 >= self.iou_threshold
        keep_2 = ious_2 >= self.iou_threshold

        keep = torch.logical_or(keep_1, keep_2)
        if keep.sum() == 0:
            # return zero loss
            return per_label_log_prob.sum() * 0
        per_label_log_prob = per_label_log_prob[keep]

        loss = -per_label_log_prob

        if weight is not None:
            weight = weight[keep]

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return loss_weight * loss

    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = self.loss_weight
        if decay_rate is not None:
            loss_weight = self.loss_weight * decay_rate
        if classes_equi is not None:
            classes_equi[nbr_classes]=nbr_classes
            if (labels_1 > nbr_classes).any():
                labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
            if (labels_2 > nbr_classes).any():
                labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

            

        if len(labels_1.shape) == 1:
            labels_1 = labels_1.reshape(-1, 1)
        if len(labels_2.shape) == 1:
            labels_2 = labels_2.reshape(-1, 1)
        if queue_trg is not None:
            if len(queue_trg.shape) == 1:
                queue_trg = queue_trg.reshape(-1, 1)

            queue_trg = queue_trg.transpose(0, 1)
            labels_1 = labels_1.reshape(-1,1)
            labels_2 = labels_2.reshape(-1,1)
            queue_trg = queue_trg.reshape(-1,1)
            
            # mask with shape [N, N], mask_{i, j}=1
            # if sample i and sample j have the same label

            labels_2 = torch.cat((labels_1, labels_2, queue_trg), dim=0)

            queue_res = torch.moveaxis(queue_res, 0,2)
            if len(queue_res.shape) > 2:
                queue_res = torch.moveaxis(queue_res, 1,2)
                queue_res = queue_res.reshape(-1, queue_res.shape[2])
            
            if len(features_2.shape) > 2:
                features_2 = torch.moveaxis(features_2, 1,2)
                features_2 = features_2.reshape(-1, features_2.shape[2])

            if len(features_1.shape) > 2:
                features_1 = torch.moveaxis(features_1, 1,2)
                features_1 = features_1.reshape(-1, features_1.shape[2])
            
           
            features_2 = torch.cat((features_1, features_2, queue_res), dim=0)
            del queue_res
        else:
            labels_2 = torch.cat((labels_1, labels_2), dim=0)

            features_2 = torch.cat((features_1, features_2), dim=0)

        

        similarityq = torch.div(torch.matmul(features_2, features_2.T), self.temperature)
        del features_2
        del features_1
        
        del labels_1
        torch.cuda.empty_cache()
        sim_row_maxq, _ = torch.max(similarityq, dim=1, keepdim=True)
        similarityq = similarityq - sim_row_maxq.detach()
        # mask out self-contrastive

        logits_mask = torch.ones_like(similarityq)
        logits_mask.fill_diagonal_(0)

        exp_simq = torch.exp(similarityq) * logits_mask
        del logits_mask

        log_probq = similarityq - torch.log(exp_simq.sum(dim=1, keepdim=True))

        del similarityq
        del exp_simq
        torch.cuda.empty_cache()
        label_maskq = (torch.eq(labels_2, labels_2.T).fill_diagonal_(0)).float().to(log_probq.device)
        del labels_2
        lbl_lengthq = label_maskq.sum(1)
        lbl_lengthq[lbl_lengthq == 0] = 1

        log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
        if queue_trg is not None:
            ious_1 = ious_1.reshape(-1, 1)
            ious_2 = ious_2.reshape(-1, 1)
            queue_iou = queue_iou.reshape(-1, 1)
            keep_queue = torch.cat((ious_1, ious_2, queue_iou), dim=0)
        else:
            keep_queue = torch.cat((ious_1, ious_2), dim=0)

        keep_queue = keep_queue.reshape(-1)
        keep_queue = keep_queue >= self.iou_threshold
        if keep_queue.sum() == 0:
            # return zero loss
            return log_probq.sum() * 0

        loss = -log_probq[keep_queue]

        if weight is not None:
            weight = weight[keep_queue]

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return loss_weight * loss

    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_light(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            if queue_trg is not None:

                if len(queue_trg.shape) == 1:
                    queue_trg = queue_trg.reshape(-1, 1)


                queue_trg = queue_trg.transpose(0, 1)
                labels_1 = labels_1.reshape(-1,1)
                labels_2 = labels_2.reshape(-1,1)
                queue_trg = queue_trg.reshape(-1,1)
                
                # mask with shape [N, N], mask_{i, j}=1
                # if sample i and sample j have the same label
                #print(f'lbl 1 {labels_1.unique()}, lbl 2 {labels_2.unique()}, queue {queue_trg.unique()}')
                labels_2 = torch.cat((labels_1, labels_2, queue_trg), dim=0)
                
                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 0,2)
                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 1,2)
                    queue_res = queue_res.reshape(-1, queue_res.shape[2])
                
                if len(features_2.shape) > 2:
                    features_2 = torch.moveaxis(features_2, 1,2)
                    features_2 = features_2.reshape(-1, features_2.shape[2])

                if len(features_1.shape) > 2:
                    features_1 = torch.moveaxis(features_1, 1,2)
                    features_1 = features_1.reshape(-1, features_1.shape[2])
                
                
                features_2 = torch.cat((features_1, features_2, queue_res), dim=0)
                del queue_res
            else:
                labels_2 = torch.cat((labels_1, labels_2), dim=0)

                features_2 = torch.cat((features_1, features_2), dim=0)

            if queue_trg is not None:
                ious_1 = ious_1.reshape(-1, 1)
                ious_2 = ious_2.reshape(-1, 1)
                queue_iou = queue_iou.reshape(-1, 1)
                keep_queue = torch.cat((ious_1, ious_2, queue_iou), dim=0)
            else:
                keep_queue = torch.cat((ious_1, ious_2), dim=0)

            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            if self.no_bg and classes_equi is not None:
                
                keep_lbl = labels_2.clone().reshape(-1) < nbr_classes
                keep_queue = torch.logical_and(keep_queue, keep_lbl)
                keep_queue = torch.logical_and(keep_queue, labels_2.clone().reshape(-1) >= 0)
            elif self.no_bg and classes_equi is None:
                keep_lbl = labels_2.clone().reshape(-1) != (nbr_classes)
                keep_queue = torch.logical_and(keep_queue, keep_lbl)
                keep_queue = torch.logical_and(keep_queue, labels_2.clone().reshape(-1) >= 0)
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            features_2 = features_2[keep_queue]
            labels_2 = labels_2[keep_queue]
            similarityq = torch.div(torch.matmul(features_2, features_2.T), self.temperature)
            del features_2
            del features_1
        
            del labels_1
            torch.cuda.empty_cache()
            sim_row_maxq, _ = torch.max(similarityq, dim=1, keepdim=True)
            similarityq = similarityq - sim_row_maxq.detach()
            # mask out self-contrastive

            logits_mask = torch.ones_like(similarityq)
            logits_mask.fill_diagonal_(0)
            exp_simq = torch.exp(similarityq) * logits_mask
            
            del logits_mask

            log_probq = similarityq - torch.log(exp_simq.sum(dim=1, keepdim=True))

            del similarityq
            del exp_simq
            torch.cuda.empty_cache()
            label_maskq = (torch.eq(labels_2, labels_2.T).fill_diagonal_(0)).float().to(log_probq.device)
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_light_bginqueue(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
                

            if queue_trg is not None:

                if len(queue_trg.shape) == 1:
                    queue_trg = queue_trg.reshape(-1, 1)


                queue_trg = queue_trg.transpose(0, 1)
                labels_1 = labels_1.reshape(-1,1)
                labels_2 = labels_2.reshape(-1,1)
                queue_trg = queue_trg.reshape(-1,1)
                
                # mask with shape [N, N], mask_{i, j}=1
                # if sample i and sample j have the same label

                labels_2 = torch.cat((labels_1, labels_2, queue_trg), dim=0)
                
                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 0,2)
                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 1,2)
                    queue_res = queue_res.reshape(-1, queue_res.shape[2])
                
                if len(features_2.shape) > 2:
                    features_2 = torch.moveaxis(features_2, 1,2)
                    features_2 = features_2.reshape(-1, features_2.shape[2])

                if len(features_1.shape) > 2:
                    features_1 = torch.moveaxis(features_1, 1,2)
                    features_1 = features_1.reshape(-1, features_1.shape[2])
                
                
                features_2 = torch.cat((features_1, features_2, queue_res), dim=0)
                del queue_res
            else:
                labels_2 = torch.cat((labels_1, labels_2), dim=0)

                features_2 = torch.cat((features_1, features_2), dim=0)

            if queue_trg is not None:
                ious_1 = ious_1.reshape(-1, 1)
                ious_2 = ious_2.reshape(-1, 1)

                queue_iou = queue_iou.reshape(-1, 1)
                keep_queue = torch.cat((ious_1, ious_2, queue_iou), dim=0)
            else:
                keep_queue = torch.cat((ious_1, ious_2), dim=0)

            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            features_2 = features_2[keep_queue]
            labels_2 = labels_2[keep_queue]
            similarityq = torch.div(torch.matmul(features_2, features_2.T), self.temperature)
            del features_2
            del features_1
        
            del labels_1
            torch.cuda.empty_cache()
            sim_row_maxq, _ = torch.max(similarityq, dim=1, keepdim=True)
            similarityq = similarityq - sim_row_maxq.detach()
            # mask out self-contrastive

            logits_mask = torch.ones_like(similarityq)
            logits_mask.fill_diagonal_(0)
            exp_simq = torch.exp(similarityq) * logits_mask
            
            del logits_mask

            log_probq = similarityq - torch.log(exp_simq.sum(dim=1, keepdim=True))

            del similarityq
            del exp_simq
            torch.cuda.empty_cache()
            label_maskq = (torch.eq(labels_2, labels_2.T).fill_diagonal_(0)).float().to(log_probq.device)
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     


@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_light_nobg(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:

                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
                

            if queue_trg is not None:

                if len(queue_trg.shape) == 1:
                    queue_trg = queue_trg.reshape(-1, 1)


                queue_trg = queue_trg.transpose(0, 1)
                labels_1 = labels_1.reshape(-1,1)
                labels_2 = labels_2.reshape(-1,1)
                queue_trg = queue_trg.reshape(-1,1)
                
                # mask with shape [N, N], mask_{i, j}=1
                # if sample i and sample j have the same label

                labels_2 = torch.cat((labels_1, labels_2, queue_trg), dim=0)
                msk_bg = labels_2.reshape(-1) < nbr_classes
                labels_2 = labels_2[msk_bg]

                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 0,2)
                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 1,2)
                    queue_res = queue_res.reshape(-1, queue_res.shape[2])
                
                if len(features_2.shape) > 2:
                    features_2 = torch.moveaxis(features_2, 1,2)
                    features_2 = features_2.reshape(-1, features_2.shape[2])

                if len(features_1.shape) > 2:
                    features_1 = torch.moveaxis(features_1, 1,2)
                    features_1 = features_1.reshape(-1, features_1.shape[2])
                
                
                features_2 = torch.cat((features_1, features_2, queue_res), dim=0)
                features_2 = features_2[msk_bg]
                del queue_res
            else:
                labels_2 = torch.cat((labels_1, labels_2), dim=0)
                msk_bg = labels_2.reshape(-1) < nbr_classes
                labels_2 = labels_2[msk_bg]

                features_2 = torch.cat((features_1, features_2), dim=0)
                features_2 = features_2[msk_bg]

            if queue_trg is not None:
                ious_1 = ious_1.reshape(-1, 1)
                ious_2 = ious_2.reshape(-1, 1)

                queue_iou = queue_iou.reshape(-1, 1)
                keep_queue = torch.cat((ious_1, ious_2, queue_iou), dim=0)
                keep_queue = keep_queue[msk_bg]
            else:
                keep_queue = torch.cat((ious_1, ious_2), dim=0)
                keep_queue = keep_queue[msk_bg]

            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            if keep_queue.sum() == 0:
                # return zero loss
                return features_2 * 0

            features_2 = features_2[keep_queue]
            labels_2 = labels_2[keep_queue]
            similarityq = torch.div(torch.matmul(features_2, features_2.T), self.temperature)
            del features_2
            del features_1
        
            del labels_1
            torch.cuda.empty_cache()
            sim_row_maxq, _ = torch.max(similarityq, dim=1, keepdim=True)
            similarityq = similarityq - sim_row_maxq.detach()
            # mask out self-contrastive

            logits_mask = torch.ones_like(similarityq)
            logits_mask.fill_diagonal_(0)
            exp_simq = torch.exp(similarityq) * logits_mask
            
            del logits_mask

            log_probq = similarityq - torch.log(exp_simq.sum(dim=1, keepdim=True))

            del similarityq
            del exp_simq
            torch.cuda.empty_cache()
            label_maskq = (torch.eq(labels_2, labels_2.T).fill_diagonal_(0)).float().to(log_probq.device)
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     


@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_light_class(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            
            labels_2 = torch.cat((labels_1, labels_2), dim=0)

            features_2 = torch.cat((features_1, features_2), dim=0)

            keep_queue = torch.cat((ious_1, ious_2), dim=0)
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            features_2 = features_2[keep_queue]
            labels_2 = labels_2[keep_queue]

            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)

            sim_row_maxq, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_class(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            labels_2 = torch.cat((labels_1, labels_2), dim=0)

            mask_background = labels_2.reshape(-1) < nbr_classes
            #labels_2 = labels_2[mask_background]
            #print(labels_2.unique(), queue_trg.unique(), labels_2.shape)
            
            features_2 = torch.cat((features_1, features_2), dim=0)
            #features_2 = features_2[mask_background]

            keep_queue_trg = queue_trg.reshape(-1)
            keep_queue_trg = keep_queue_trg >= 0

            queue_res = queue_res[keep_queue_trg]
            queue_trg = queue_trg[keep_queue_trg]
            #print(queue_trg.unique(), labels_2.unique())
            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)

            #print(dict().shape)
            sim_row_maxq, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            #print(f'unique lbl {labels_2.unique()} trf queue {queue_trg.unique()}')
            
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            keep_queue = torch.cat((ious_1, ious_2), dim=0)
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            log_probq = log_probq[keep_queue]
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_class_weighted(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            labels_2 = torch.cat((labels_1, labels_2), dim=0)

            mask_background = labels_2.reshape(-1) < nbr_classes
            #labels_2 = labels_2[mask_background]
            #print(labels_2.unique(), queue_trg.unique(), labels_2.shape)
            
            features_2 = torch.cat((features_1, features_2), dim=0)
            #features_2 = features_2[mask_background]

            keep_queue_trg = queue_trg.reshape(-1)
            keep_queue_trg = keep_queue_trg >= 0

            queue_res = queue_res[keep_queue_trg]
            queue_trg = queue_trg[keep_queue_trg]

            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)

            sim_row_maxq, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            #print(f'unique lbl {labels_2.unique()} trf queue {queue_trg.unique()}')
            
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            keep_queue = torch.cat((ious_1, ious_2), dim=0)
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            log_probq = log_probq[keep_queue]
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     


@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_class_nobg(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            labels_2 = torch.cat((labels_1, labels_2), dim=0)
            
            mask_background = labels_2.reshape(-1) < nbr_classes
            labels_2 = labels_2[mask_background]
            
            
            features_2 = torch.cat((features_1, features_2), dim=0)
            features_2 = features_2[mask_background]

            keep_queue_trg = queue_trg.reshape(-1)
            keep_queue_trg = keep_queue_trg >= 0

            queue_res = queue_res[keep_queue_trg]
            queue_trg = queue_trg[keep_queue_trg]
            
            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)

            sim_row_maxq, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            keep_queue = torch.cat((ious_1, ious_2), dim=0)
            keep_queue = keep_queue[mask_background]
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            log_probq = log_probq[keep_queue]
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_class_withbgqueue(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                #print(f'class eq {classes_equi}')
                classes_equi[nbr_classes]=nbr_classes
                #print(f'lbl 1 {labels_1.unique()} lbl 2 {labels_2.unique()}')
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            labels_2 = torch.cat((labels_1, labels_2), dim=0)
            #print(labels_2.unique(), queue_trg.unique(), labels_2.shape)
            mask_background = labels_2.reshape(-1) < nbr_classes
            labels_2 = labels_2[mask_background]
            
            
            features_2 = torch.cat((features_1, features_2), dim=0)
            features_2 = features_2[mask_background]

            keep_queue_trg = queue_trg.reshape(-1)

            keep_queue_trg = keep_queue_trg >= 0
            queue_res = queue_res[keep_queue_trg]
            queue_trg = queue_trg[keep_queue_trg]
            print(queue_trg)
            sim = torch.div(torch.matmul(queue_res.clone(), features_2.T), self.temperature)
            print(f'sim shape T {sim.shape}')
            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)
            print(f'sim shape {sim.shape}')
            sim_row_maxq, sim_row_maxid = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            #print(f'unique lbl {labels_2.unique()} trf queue {queue_trg.unique()}')
            
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            keep_queue = torch.cat((ious_1, ious_2), dim=0)
            keep_queue = keep_queue[mask_background]
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            log_probq = log_probq[keep_queue]
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_class_withbgqueue_all(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                features_3: Tensor,
                labels_3: Tensor,
                ious_3: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                #print(f'class eq {classes_equi}')
                classes_equi[nbr_classes]=nbr_classes
                #print(f'lbl 1 {labels_1.unique()} lbl 2 {labels_2.unique()}')
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)
                if (labels_3 > nbr_classes).any():
                    labels_3 = torch.tensor([classes_equi[int(i)] for i in labels_3]).to(features_3.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            if len(labels_3.shape) == 1:
                labels_3 = labels_3.reshape(-1, 1)
            labels_2 = torch.cat((labels_1, labels_2, labels_3), dim=0)
            #print(labels_2.unique(), queue_trg.unique(), labels_2.shape)
            mask_background = labels_2.reshape(-1) < nbr_classes
            labels_2 = labels_2[mask_background]
            
            
            features_2 = torch.cat((features_1, features_2, features_3), dim=0)
            features_2 = features_2[mask_background]

            #keep_queue_trg = queue_trg.reshape(-1)

            #keep_queue_trg = keep_queue_trg >= 0
            #queue_res = queue_res[keep_queue_trg]
            #queue_trg = queue_trg[keep_queue_trg]

            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)

            sim_row_maxq, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            #print(f'unique lbl {labels_2.unique()} trf queue {queue_trg.unique()}')
            
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            keep_queue = torch.cat((ious_1, ious_2, ious_3), dim=0)
            keep_queue = keep_queue[mask_background]
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            log_probq = log_probq[keep_queue]
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     


@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_class_withbg(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=-1
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            labels_2 = torch.cat((labels_1, labels_2), dim=0)
            
            labels_2[labels_2 == nbr_classes] = -1
            
            
            features_2 = torch.cat((features_1, features_2), dim=0)
            #features_2 = features_2[mask_background]

            #keep_queue_trg = queue_trg.reshape(-1)
            #keep_queue_trg = keep_queue_trg >= 0

            #queue_trg = queue_trg[keep_queue_trg]
            #queue_res = queue_res[keep_queue_trg]

            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)
            
            sim_row_maxq, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            keep_queue = torch.cat((ious_1, ious_2), dim=0)
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            log_probq = log_probq[keep_queue]
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     


@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_class_all(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                features_3: Tensor,
                labels_3: Tensor,
                ious_3: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)
                if (labels_3 > nbr_classes).any():
                    labels_3 = torch.tensor([classes_equi[int(i)] for i in labels_3]).to(features_3.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            if len(labels_3.shape) == 1:
                labels_3 = labels_3.reshape(-1, 1)
            
            labels_2 = torch.cat((labels_1, labels_2, labels_3), dim=0)

            features_2 = torch.cat((features_1, features_2, features_3), dim=0)

            keep_queue_trg = queue_trg.reshape(-1)
            keep_queue_trg = keep_queue_trg >= 0

            queue_res = queue_res[keep_queue_trg]
            queue_trg = queue_trg[keep_queue_trg]

            sim = torch.div(torch.matmul(features_2, queue_res.clone().T), self.temperature)

            sim_row_maxq, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_row_maxq.detach()
            esim = torch.exp(sim)
            log_probq = sim - torch.log(esim.sum(dim=1, keepdim=True))

            label_maskq = (torch.eq(labels_2, queue_trg.clone().T)).float().to(log_probq.device)
            
            
            del sim
            del esim
            torch.cuda.empty_cache()
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            keep_queue = torch.cat((ious_1, ious_2, ious_3), dim=0)
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            log_probq = log_probq[keep_queue]
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     


@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_light_classif(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                

            #if len(labels_1.shape) == 1:
            #    labels_1 = labels_1.reshape(-1, 1)

            keep_queue = ious_1.clone()
            keep_queue = keep_queue >= self.iou_threshold
            
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            features_1 = features_1[keep_queue]
            labels_1 = labels_1[keep_queue]

            sim = torch.div(torch.matmul(features_1, queue_res.clone().T), self.temperature)

            CEsim = torch.nn.functional.softmax(sim, dim=1)
            loss = F.cross_entropy(
                CEsim,
                labels_1,
                reduction='none')

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss, CEsim
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     

      

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_light_all(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False,
                 no_bg = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss
        self.no_bg = no_bg

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                features_3: Tensor,
                labels_3: Tensor,
                ious_3: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)
                if (labels_3 > nbr_classes).any():
                    labels_3 = torch.tensor([classes_equi[int(i)] for i in labels_3]).to(features_3.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            if len(labels_3.shape) == 1:
                labels_3 = labels_3.reshape(-1, 1)
            if queue_trg is not None:
                if len(queue_trg.shape) == 1:
                    queue_trg = queue_trg.reshape(-1, 1)

                queue_trg = queue_trg.transpose(0, 1)
                labels_1 = labels_1.reshape(-1,1)
                labels_2 = labels_2.reshape(-1,1)
                labels_3 = labels_3.reshape(-1,1)
                queue_trg = queue_trg.reshape(-1,1)
                
                # mask with shape [N, N], mask_{i, j}=1
                # if sample i and sample j have the same label

                labels_2 = torch.cat((labels_1, labels_2, labels_3, queue_trg), dim=0)

                queue_res = torch.moveaxis(queue_res, 0,2)
                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 1,2)
                    queue_res = queue_res.reshape(-1, queue_res.shape[2])
                
                if len(features_3.shape) > 2:
                    features_3 = torch.moveaxis(features_3, 1,2)
                    features_3 = features_3.reshape(-1, features_3.shape[2])
                
                if len(features_2.shape) > 2:
                    features_2 = torch.moveaxis(features_2, 1,2)
                    features_2 = features_2.reshape(-1, features_2.shape[2])

                if len(features_1.shape) > 2:
                    features_1 = torch.moveaxis(features_1, 1,2)
                    features_1 = features_1.reshape(-1, features_1.shape[2])
                
                
                features_2 = torch.cat((features_1, features_2, features_3, queue_res), dim=0)
                del queue_res
            else:
                labels_2 = torch.cat((labels_1, labels_2, labels_3), dim=0)

                features_2 = torch.cat((features_1, features_2, features_3), dim=0)
            del features_1
            del features_3
            if queue_trg is not None:
                ious_1 = ious_1.reshape(-1, 1)
                ious_2 = ious_2.reshape(-1, 1)
                ious_3 = ious_3.reshape(-1, 1)
                queue_iou = queue_iou.reshape(-1, 1)
                keep_queue = torch.cat((ious_1, ious_2, ious_3, queue_iou), dim=0)
            else:
                keep_queue = torch.cat((ious_1, ious_2, ious_3), dim=0)

            del ious_1
            del ious_2
            del ious_3
            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            if self.no_bg:
                keep_lbl = labels_2.clone().reshape(-1) < 20
                keep_queue = torch.logical_and(keep_queue, keep_lbl)
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            
            features_2 = features_2[keep_queue]
            labels_2 = labels_2[keep_queue]
            
            similarityq = torch.div(torch.matmul(features_2, features_2.T), self.temperature)
            del features_2
            
        
            del labels_1
            torch.cuda.empty_cache()
            sim_row_maxq, _ = torch.max(similarityq, dim=1, keepdim=True)
            similarityq = similarityq - sim_row_maxq.detach()
            # mask out self-contrastive

            logits_mask = torch.ones_like(similarityq)
            logits_mask.fill_diagonal_(0)

            exp_simq = torch.exp(similarityq) * logits_mask
            del logits_mask

            log_probq = similarityq - torch.log(exp_simq.sum(dim=1, keepdim=True))

            del similarityq
            del exp_simq
            torch.cuda.empty_cache()
            label_maskq = (torch.eq(labels_2, labels_2.T).fill_diagonal_(0)).float().to(log_probq.device)
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     
              
           

@LOSSES.register_module()
class QueueDualSupervisedContrastiveLoss_light_Neg(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0,
                 no_loss = False) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_loss = no_loss

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if not self.no_loss:
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_weight = self.loss_weight
            if decay_rate is not None:
                loss_weight = self.loss_weight * decay_rate
            if classes_equi is not None:
                classes_equi[nbr_classes]=nbr_classes
                if (labels_1 > nbr_classes).any():
                    labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
                if (labels_2 > nbr_classes).any():
                    labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

                

            if len(labels_1.shape) == 1:
                labels_1 = labels_1.reshape(-1, 1)
            if len(labels_2.shape) == 1:
                labels_2 = labels_2.reshape(-1, 1)
            if queue_trg is not None:
                if len(queue_trg.shape) == 1:
                    queue_trg = queue_trg.reshape(-1, 1)

                queue_trg = queue_trg.transpose(0, 1)
                labels_1 = labels_1.reshape(-1,1)
                labels_2 = labels_2.reshape(-1,1)
                queue_trg = queue_trg.reshape(-1,1)
                
                # mask with shape [N, N], mask_{i, j}=1
                # if sample i and sample j have the same label

                labels_2 = torch.cat((labels_1, labels_2, queue_trg), dim=0)

                queue_res = torch.moveaxis(queue_res, 0,2)
                if len(queue_res.shape) > 2:
                    queue_res = torch.moveaxis(queue_res, 1,2)
                    queue_res = queue_res.reshape(-1, queue_res.shape[2])
                
                if len(features_2.shape) > 2:
                    features_2 = torch.moveaxis(features_2, 1,2)
                    features_2 = features_2.reshape(-1, features_2.shape[2])

                if len(features_1.shape) > 2:
                    features_1 = torch.moveaxis(features_1, 1,2)
                    features_1 = features_1.reshape(-1, features_1.shape[2])
                
            
                features_2 = torch.cat((features_1, features_2, queue_res), dim=0)

                del queue_res
            else:
                labels_2 = torch.cat((labels_1, labels_2), dim=0)

                features_2 = torch.cat((features_1, features_2), dim=0)

            if queue_trg is not None:
                ious_1 = ious_1.reshape(-1, 1)
                ious_2 = ious_2.reshape(-1, 1)
                queue_iou = queue_iou.reshape(-1, 1)
                keep_queue = torch.cat((ious_1, ious_2, queue_iou), dim=0)
            else:
                keep_queue = torch.cat((ious_1, ious_2), dim=0)

            keep_queue = keep_queue.reshape(-1)
            keep_queue = keep_queue >= self.iou_threshold
            if keep_queue.sum() == 0:
                # return zero loss
                return log_probq.sum() * 0

            
            features_2 = features_2[keep_queue]
            labels_2 = labels_2[keep_queue]

            print(f'features norm: {torch.norm(features_2)}')

            similarityq = torch.div(torch.matmul(features_2, features_2.T), self.temperature)
            del features_2
            del features_1
        
            del labels_1
            torch.cuda.empty_cache()
            sim_row_maxq, _ = torch.max(similarityq, dim=1, keepdim=True)
            similarityq = similarityq - sim_row_maxq.detach()
            # mask out self-contrastive

            logits_mask = torch.ones_like(similarityq)
            logits_mask.fill_diagonal_(0)

            exp_simq = torch.exp(similarityq) * logits_mask
            del logits_mask

            log_probq = similarityq - torch.log(exp_simq.sum(dim=1, keepdim=True))

            del similarityq
            del exp_simq
            torch.cuda.empty_cache()
            label_maskq = (torch.eq(labels_2, labels_2.T).fill_diagonal_(0)).float().to(log_probq.device)
            del labels_2
            lbl_lengthq = label_maskq.sum(1)
            lbl_lengthq[lbl_lengthq == 0] = 1

            log_probq = (log_probq * label_maskq).sum(1) / lbl_lengthq
            

            loss = -log_probq

            if weight is not None:
                weight = weight[keep_queue]

            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

            return loss_weight * loss
        else:
            return -torch.ones(1)
    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     
            
@LOSSES.register_module()
class QueueDualSupervisedContrastiveLossBU(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    def __init__(self,
                 temperature: float = 0.2,
                 iou_threshold: float = 0.5,
                 reweight_type: Literal['none', 'exp', 'linear'] = 'none',
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = self._get_reweight_func(reweight_type)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                features_1: Tensor,
                labels_1: Tensor,
                ious_1: Tensor,
                features_2: Tensor,
                labels_2: Tensor,
                ious_2: Tensor,
                queue_res: Tensor = None, 
                queue_trg: Tensor = None, 
                queue_iou: Tensor = None,
                classes_equi = None,
                nbr_classes = None,
                decay_rate: Optional[float] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            features (tensor): Shape of (N, K) where N is the number
                of features to be compared and K is the channels.
            labels (tensor): Shape of (N).
            ious (tensor): Shape of (N).
            queue features (tensor): Shape of Qx(N, K) 
                where N is the number of features to be compared 
                K is the channels
                Q is the queue length
            queue labels (tensor): Shape of Qx(N).
            queue ious (tensor): Shape of Qx(N).
            decay_rate (float | None): The decay rate for total loss.
                Default: None.
            weight (Tensor | None): The weight of loss for each
                prediction with shape of (N). Default: None.
            avg_factor (int | None): Average factor that is used to average
                the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = self.loss_weight
        if decay_rate is not None:
            loss_weight = self.loss_weight * decay_rate
        if classes_equi is not None:
            classes_equi[nbr_classes]=nbr_classes
            if (labels_1 > nbr_classes).any():
                labels_1 = torch.tensor([classes_equi[int(i)] for i in labels_1]).to(features_1.device)
            if (labels_2 > nbr_classes).any():
                labels_2 = torch.tensor([classes_equi[int(i)] for i in labels_2]).to(features_2.device)

            

        if len(labels_1.shape) == 1:
            labels_1 = labels_1.reshape(-1, 1)
        if len(labels_2.shape) == 1:
            labels_2 = labels_2.reshape(-1, 1)
        if queue_trg is not None:
            if len(queue_trg.shape) == 1:
                queue_trg = queue_trg.reshape(-1, 1)

            queue_trg = queue_trg.transpose(0, 1)
            labels_1 = labels_1.reshape(-1,1)
            labels_2 = labels_2.reshape(-1,1)
            queue_trg = queue_trg.reshape(-1,1)
           # print(f'shape {labels_1.shape, labels_2.shape, queue_trg.shape}')

            #labels_2 = torch.cat((labels_1, labels_2, queue_trg), dim=1)
            
            # mask with shape [N, N], mask_{i, j}=1
            # if sample i and sample j have the same label
            #labels_2 = labels_2.reshape(-1, 1)
            labels_2 = torch.cat((labels_1, labels_2, queue_trg), dim=0)

            queue_res = torch.moveaxis(queue_res, 0,2)
            if len(queue_res.shape) > 2:
                queue_res = torch.moveaxis(queue_res, 1,2)
                queue_res = queue_res.reshape(-1, queue_res.shape[2])
            
            if len(features_2.shape) > 2:
                features_2 = torch.moveaxis(features_2, 1,2)
                features_2 = features_2.reshape(-1, features_2.shape[2])

            if len(features_1.shape) > 2:
                features_1 = torch.moveaxis(features_1, 1,2)
                features_1 = features_1.reshape(-1, features_1.shape[2])
            
            #features_2 = torch.cat((torch.unsqueeze(features_1, dim=2), torch.unsqueeze(features_2, dim=2), queue_res), dim=2)
           # print(f'shape {features_1.shape, features_2.shape, queue_res.shape}')

            features_2 = torch.cat((features_1, features_2, queue_res), dim=0)
           # print(f'shape {features_2.shape}')
           # print('======')
            #features_2 = torch.moveaxis(features_2, 2, 1)

            #features_2 = features_2.reshape(-1, features_2.shape[2])
            del queue_res
        else:
           # if labels_1.shape[0] == labels_2.shape[0]:
            labels_2 = torch.cat((labels_1, labels_2), dim=0)
                
           # elif labels_1.shape[1] == labels_2.shape[1]:
           #     labels_2 = torch.cat((labels_1, labels_2), dim=0)
                
                # mask with shape [N, N], mask_{i, j}=1
                # if sample i and sample j have the same label
           #     labels_2 = labels_2.reshape(-1, 1)
           # else:
           #     labels_2 = torch.cat((torch.unsqueeze(labels_1, dim=2), torch.unsqueeze(labels_2, dim=2)), dim=2)
                
                # mask with shape [N, N], mask_{i, j}=1
                # if sample i and sample j have the same label
           #     labels_2 = labels_2.reshape(-1, 1)

            features_2 = torch.cat((features_1, features_2), dim=0)
            #features_2 = torch.moveaxis(features_2, 2, 1)

            #features_2 = features_2.reshape(-1, features_2.shape[2])

        similarityq = torch.div(torch.matmul(features_2, features_2.T), self.temperature)
        del features_2
        del features_1
        
        del labels_1
        torch.cuda.empty_cache()
        sim_row_maxq, _ = torch.max(similarityq, dim=1, keepdim=True)
        similarityq = similarityq - sim_row_maxq.detach()
        # mask out self-contrastive
        sshapes= similarityq.shape
        logits_mask = torch.ones_like(similarityq)
        logits_mask.fill_diagonal_(0)

        exp_simq = torch.exp(similarityq) * logits_mask
        log_probq = similarityq - torch.log(exp_simq.sum(dim=1, keepdim=True))
        del similarityq
        del exp_simq
        torch.cuda.empty_cache()
        label_maskq = torch.eq(labels_2, labels_2.T).float().to(log_probq.device)
        del labels_2
        lbl_lengthq = label_maskq.sum(1)
        lbl_lengthq[lbl_lengthq == 0] = 1
        #per_label_log_probq = (log_probq * logits_mask.to(log_probq.get_device()) * label_maskq).sum(1) / lbl_lengthq
        print(f'shapes {log_probq.shape, logits_mask.shape, label_maskq.shape}')
        log_probq = (log_probq * logits_mask.to(log_probq.get_device()) * label_maskq).sum(1) / lbl_lengthq
        
        if queue_trg is not None:
            ious_1 = ious_1.reshape(-1, 1)
            ious_2 = ious_2.reshape(-1, 1)
            queue_iou = queue_iou.reshape(-1, 1)
            #keep_queue = torch.cat((torch.unsqueeze(ious_1, dim=0), torch.unsqueeze(ious_2, dim=0), queue_iou), dim=0)
            keep_queue = torch.cat((ious_1, ious_2, queue_iou), dim=0)
        else:
            keep_queue = torch.cat((ious_1, ious_2), dim=0)

        keep_queue = keep_queue.reshape(-1)
        keep_queue = keep_queue >= self.iou_threshold
        if keep_queue.sum() == 0:
            # return zero loss
            #return per_label_log_probq.sum() * 0
            return log_probq.sum() * 0
        #per_label_log_probq = per_label_log_probq[keep_queue]
        log_probq = log_probq[keep_queue]

        #loss = -per_label_log_probq
        loss = -log_probq

        if weight is not None:
            weight = weight[keep_queue]

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return loss_weight * loss

    @staticmethod
    def _get_reweight_func(
            reweight_type: Literal['none', 'exp',
                                   'linear'] = 'none') -> callable:
        """Return corresponding reweight function according to `reweight_type`.

        Args:
            reweight_type (str): Reweight function for contrastive loss.
                Options are ('none', 'exp', 'linear'). Default: 'none'.

        Returns:
            callable: Used for reweight loss.
        """
        assert reweight_type in ('none', 'exp', 'linear'), \
            f'not support `reweight_type` {reweight_type}.'
        if reweight_type == 'none':

            def trivial(iou):
                return torch.ones_like(iou)

            return trivial
        elif reweight_type == 'linear':

            def linear(iou):
                return iou

            return linear
        elif reweight_type == 'exp':

            def exp_decay(iou):
                return torch.exp(iou) - 1

            return exp_decay     
                           


from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss



# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          threshold_reduce=0.5,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    #print(pred.shape)
    pred_sigmoid = pred.sigmoid()
    #print(pred_sigmoid.shape)
    target = F.one_hot(target, num_classes=pred.shape[1]).type_as(pred)

    ce = target*(-torch.log(pred_sigmoid))
    w = target*(((1 - pred_sigmoid)/threshold_reduce).pow(gamma))
    msk = pred_sigmoid < threshold_reduce
    w[msk] = 1
    #print(ce.shape, w.shape, msk.shape)
    loss = alpha*(w*ce)
    

    #loss = F.cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class ReducedFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(ReducedFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls

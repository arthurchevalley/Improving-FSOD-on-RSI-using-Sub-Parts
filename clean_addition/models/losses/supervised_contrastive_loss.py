# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmdet.models import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from torch import Tensor
from typing_extensions import Literal


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
        per_label_log_probq = (log_probq * logits_mask.to(log_probq.get_device()) * label_maskq).sum(1) / lbl_lengthq
        
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
            return per_label_log_probq.sum() * 0
        per_label_log_probq = per_label_log_probq[keep_queue]

        loss = -per_label_log_probq

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
            
                       
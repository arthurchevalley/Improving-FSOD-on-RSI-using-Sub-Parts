# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from .convfc_bbox_head import Shared2FCBBoxHeadUpdate
from mmdet.core import multi_apply


@HEADS.register_module()
class MPSR_AugContrastiveBBoxHead_Branch(Shared2FCBBoxHeadUpdate):
    """BBoxHead for `FSCE <https://arxiv.org/abs/2103.05950>`_.

    Args:
        mlp_head_channels (int): Output channels of contrast branch
            mlp. Default: 128.
        with_weight_decay (bool): Whether to decay loss weight. Default: False.
        loss_contrast (dict): Config of contrast loss.
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bo ol): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    def __init__(self,
                 auxiliary_loss_weight: float = 0.1,
                 mlp_head_channels: int = 128,
                 loss_cosine: Dict = dict(
                     type='CosineSim',
                     margin = .1,
                     max_contrastive_loss = 3., 
                     loss_weight=.01),
                 loss_c_cls: Dict = dict(
                     type='CosineSim',
                     margin = .1,
                     max_contrastive_loss = 3., 
                     loss_weight=.01),
                 scale: int = 20,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 inplace_relu = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.auxiliary_avg_pooling = nn.AdaptiveAvgPool2d(self.roi_feat_size)
        assert auxiliary_loss_weight >= 0
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # override the fc_cls in :obj:`Shared2FCBBoxHead`
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
            
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))

        self.contrast_loss_cls = build_loss(loss_c_cls)

        self.contrast_loss = build_loss(loss_cosine)

    def forward(self, base_x, x, x_aug):
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
                contrast_feat (Tensor): Box features for contrast loss,
                    has shape (num_proposals, C).
        """
        if x is not None and x_aug is not None:
            cls_score, bbox_pred, contrast_feat  = self.forward_one_branch(x)
            aug_cls_score, aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug)
        else:
            cls_score, bbox_pred, aug_cls_score, aug_bbox_pred = None, None, None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_cls_score, base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_cls_score, base_bbox_pred, base_contrast_feat, cls_score, bbox_pred, contrast_feat, aug_cls_score, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x):
        # shared par
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_contra = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # contrastive branch
        contrast_feat = self.contrastive_head(x_contra)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        return cls_score, bbox_pred, contrast_feat

    def forward_auxiliary_single(self, x):
        """Forward function for auxiliary of single scale."""
        x = self.auxiliary_avg_pooling(x)
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for i, fc in enumerate(self.cls_fcs):
            if (i + 1) == len(self.cls_fcs):
                x_cls = fc(x_cls)
            else:
                x_cls = self.relu(fc(x_cls))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        return cls_score,

    def forward_auxiliary(self, x):
        """Forward auxiliary features at multiple scales.

        Args:
            x (tuple[Tensor]): List of features at multiple scales, each
                is a 4D-tensor.

        Returns:
            tuple[Tensor]: Classification scores for all scale levels, each is
                a 4D-tensor, the channels number is num_anchors * num_classes.
        """
        return multi_apply(self.forward_auxiliary_single, x)

    @force_fp32(apply_to=('cls_score'))
    def auxiliary_loss(self,
                       cls_score,
                       labels,
                       label_weights,
                       reduction_override = None):
        """Compute loss for auxiliary features.

        Args:
            cls_score (Tensor): Classification scores for all scales with
                shape (num_proposals, num_classes).
            labels (Tensor): Labels of each proposal with shape
                (num_proposals).
            label_weights (Tensor): Label weights of each proposal with shape
                (num_proposals).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:
            loss_cls_ = self.auxiliary_loss_weight * self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['loss_cls_auxiliary'] = loss_cls_
            losses['acc_auxiliary'] = accuracy(cls_score, labels)
        return losses

    def set_decay_rate(self, decay_rate: float) -> None:
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.

        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

    @force_fp32(apply_to=('cont_feat'))
    def loss_contrast(self,
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_pred,
                    bbox_results,
                    gt_base_bbox,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    transform_applied,
                    num_classes,
                    reduction_override: Optional[str] = None) -> Dict:
        """Loss for contract.

        Args:
            contrast_feat (tensor): BBox features with shape (N, C)
                used for contrast loss.
            proposal_ious (tensor): IoU between proposal and ground truth
                corresponding to each BBox features with shape (N).
            labels (tensor): Labels for each BBox features with shape (N).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Dict: The calculated loss.
        """

        losses = dict()
        
        gt_nlabels_tmp = []
        gt_labels_aug_tmp = []

        for i in range(len(gt_labels_aug_true)):
            gt_labels_aug_tmp += gt_labels_aug_true[i].tolist()
            gt_nlabels_tmp += gt_nlabels[i].tolist()
        
        classes_eq = {gt_nlabels_tmp[i]: gt_labels_aug_tmp[i] for i in range(len(gt_nlabels_tmp))}
        

        losses = dict()

        losses['loss_cosine'] = self.contrast_loss(
            base_bbox_pred['cont_feat'], 
            bbox_results['cont_feat'], 
            gt_base_bbox[0], 
            bbox_targets[0],
            classes_eq, 
            num_classes,
            reduction_override=reduction_override)


        losses['loss_c_cls'] = self.contrast_loss_cls(
            bbox_results['cont_feat'], 
            aug_bbox_results['cont_feat'], 
            bbox_targets[0], 
            aug_bbox_targets[0], 
            None, 
            num_classes,
            reduction_override=reduction_override)

        return losses
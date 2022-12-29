# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from .convfc_bbox_head import Shared2FCBBoxHeadUpdate


@HEADS.register_module()
class AugContrastiveBBoxHead(Shared2FCBBoxHeadUpdate):
    """BBoxHead for `FSCE <https://arxiv.org/abs/2103.05950>`_.

    Args:
        mlp_head_channels (int): Output channels of contrast branch
            mlp. Default: 128.
        with_weight_decay (bool): Whether to decay loss weight. Default: False.
        loss_contrast (dict): Config of contrast loss.
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bool): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    def __init__(self,
                 loss_cosine=None,
                 loss_c_bbox: Dict =dict(
                     type='ConstellationLossBBOX', 
                     beta = 1.0/9.0,
                     K=2, 
                     max_contrastive_loss = 2, 
                     loss_weight=.05),
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
        

        self.contrast_loss_bbox = build_loss(loss_c_bbox)
        self.contrast_loss_cls = build_loss(loss_c_cls)
        if loss_cosine is not None:
            self.contrast_loss = build_loss(loss_cosine)
        else:
            self.contrast_loss = None
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
            cls_score, bbox_pred = self.forward_one_branch(x)
            aug_cls_score, aug_bbox_pred = self.forward_one_branch(x_aug)
        else:
            cls_score, bbox_pred, aug_cls_score, aug_bbox_pred = None, None, None, None
            
        base_cls_score, base_bbox_pred = self.forward_one_branch(base_x)
        
        return base_cls_score, base_bbox_pred, cls_score, bbox_pred, aug_cls_score, aug_bbox_pred 


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

        return cls_score, bbox_pred



    def set_decay_rate(self, decay_rate: float) -> None:
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.

        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
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
        if self.contrast_loss is not None:
            losses['loss_cosine'] = self.contrast_loss(
                base_bbox_pred['cls_score'], 
                bbox_results['cls_score'], 
                gt_base_bbox[0], 
                bbox_targets[0],
                classes_eq, 
                num_classes,
                reduction_override=reduction_override)

        
        losses['loss_c_cls'] = self.contrast_loss_cls(
            bbox_results['cls_score'], 
            aug_bbox_results['cls_score'], 
            bbox_targets[0], 
            aug_bbox_targets[0], 
            None, 
            num_classes,
            reduction_override=reduction_override)


        pos_inds_aug = (aug_bbox_targets[0] >= 0) & (aug_bbox_targets[0] > self.num_classes) 
        pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] > self.num_classes)
        
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any() and pos_inds_aug.any():
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_score_pred.view(
                    bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
            else:
                bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                pos_bbox_pred = bbox_score_pred.view(
                    bbox_score_pred.shape[0], -1, 4)[pos_inds.type(torch.bool),labels_true]
            
            if self.reg_class_agnostic:
                aug_pos_bbox_pred = aug_bbox_score_pred.view(
                    aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
            else:
                bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                aug_pos_bbox_pred = aug_bbox_score_pred.view(
                    aug_bbox_score_pred.shape[0], -1,
                    4)[pos_inds_aug.type(torch.bool),labels_true_aug]
            
            min_size = min(aug_pos_bbox_pred.shape[0], pos_bbox_pred.shape[0])
        losses['loss_c_bbox'] = self.contrast_loss_bbox(
            bbox_label,
            bbox_label_aug,
            labels_true_aug,
            labels_true,
            aug_pos_bbox_pred,
            pos_bbox_pred, 
            bbox_targets[2][pos_inds.type(torch.bool)],
            aug_bbox_targets[2][pos_inds_aug.type(torch.bool)],
            transform_applied,
            gt_labels_aug,
            min_size,
            base_bbox_pred,
            gt_base_bbox,
            reduction_override=reduction_override)
        return losses
# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from .convfc_bbox_head import Shared2FCBBoxHead
import numpy as np
import pickle 

@HEADS.register_module()
class AugContrastiveBBoxHead_Branch(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 to_norm_cls = False,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_c_cls is not None:
            self.contrast_loss_cls = build_loss(loss_c_cls)
        else:
            self.contrast_loss_cls = None

        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None

        if loss_cosine is not None:
            self.contrast_loss = build_loss(loss_cosine)
        else:
            self.contrast_loss = None
        if loss_base_aug is not None:
            self.contrast_loss_base_aug = build_loss(loss_base_aug)
        else:
            self.contrast_loss_base_aug = None
 
        self.to_norm = to_norm_cls
        

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

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if self.to_norm:
        # cls branch
            if x_cls.dim() > 2:
                x_cls = torch.flatten(x_cls, start_dim=1)

            # normalize the input x along the `input_size` dimension
            x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_cls_normalized = x_cls.div(x_norm + self.eps)
            # normalize weight
            with torch.no_grad():
                temp_norm = torch.norm(self.fc_cls.weight, p=2,dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
                self.fc_cls.weight.div_(temp_norm + self.eps)
            # calculate and scale cls_score
            cls_score = self.scale * self.fc_cls(x_cls_normalized) if self.with_cls else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None

        # contrastive branch
        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)

        return cls_score, bbox_pred, cont_feat

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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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
        

        if base_proposal_ious is not None:

            if self.with_weight_decay:
                decay_rate = self._decay_rate
            if self.contrast_loss_base_aug is not None:
                losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
                    
            if self.contrast_loss_cls is not None:
                losses['loss_c_cls'] = self.contrast_loss_cls(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss is not None:
                losses['loss_cosine'] = self.contrast_loss(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
        else:

            if self.contrast_loss is not None:
                losses['loss_cosine'] = self.contrast_loss(
                    base_bbox_pred['cont_feat'], 
                    bbox_results['cont_feat'], 
                    gt_base_bbox[0], 
                    bbox_targets[0],
                    classes_eq, 
                    num_classes,
                    reduction_override=reduction_override)
            if self.contrast_loss_cls is not None:
                losses['loss_c_cls'] = self.contrast_loss_cls(
                    bbox_results['cont_feat'], 
                    aug_bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    aug_bbox_targets[0], 
                    None, 
                    num_classes,
                    reduction_override=reduction_override)

        

        if self.contrast_loss_bbox is not None:
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

@HEADS.register_module()
class QueueAugContrastiveBBoxHead_Branch(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 loss_all = None,
                 contrast_loss_classif = None,
                 to_norm_cls = False,
                 main_training = False,
                 same_class = False,
                 same_class_all = False,
                 queue_path = 'init_queue.p',
                 use_queue = False,
                 use_base_queue = True,
                 use_novel_queue = True,
                 use_aug_queue = True,
                 queue_length = 60,
                 num_proposals = 256,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_all is None:
            if loss_c_cls is not None:
                self.contrast_loss_cls = build_loss(loss_c_cls)
            else:
                self.contrast_loss_cls = None

            if loss_cosine is not None:
                self.contrast_loss = build_loss(loss_cosine)
            else:
                self.contrast_loss = None
            if loss_base_aug is not None:
                self.contrast_loss_base_aug = build_loss(loss_base_aug)
            else:
                self.contrast_loss_base_aug = None
            if contrast_loss_classif is not None:
                self.contrast_loss_classif = build_loss(contrast_loss_classif)
            else:
                self.contrast_loss_classif = None
            self.contrast_loss_all = None
        else:
            self.contrast_loss_all = build_loss(loss_all)
            self.contrast_loss_cls = None
            self.contrast_loss = None
            self.contrast_loss_base_aug = None


        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.save_bbox_feat = False
        self.num_proposals = num_proposals
        self.queue_length = queue_length
        if self.use_queue:
            queue_base_res, queue_base_trg, queue_base_iou = self.load_queue_3(use_base_queue, use_novel_queue, use_aug_queue)
            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_trgX', queue_base_trg)

            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('queue_ptr_X', torch.zeros(1, dtype=torch.long))

        else:
            self.queue_res = None

        self.to_norm = to_norm_cls

        self.same_class = same_class
        self.same_class_all = same_class_all
        self.main_training = main_training

        self.id_save_FT = 1
            
    def queue_init(self, new_dict):
        if self.queue_res is None:
            self.queue_res = new_dict
        else:
            z = self.queue_res.copy()   
            z.update(new_dict)
            self.queue_res = z

        with open(self.queue_path, 'wb') as fp:
            pickle.dump(self.queue_res, fp, protocol=pickle.HIGHEST_PROTOCOL)


    @torch.no_grad()
    def update_queue(self, new_base, new_novel, new_aug):

        # new X: feat, lbl, iou 
        #   Feat : Ql x N x D
        #   lbl : Ql x N
        #   iou : Ql x N x D


        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        queue_length_frac = int(queue_base_res.shape[0]/3)
        
        bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1

        if queue_base_res.shape[0] < self.queue_length:

            queue_new_res = torch.cat((queue_base_res[:queue_length_frac], torch.unsqueeze(new_base[0], dim=0), 
                                        queue_base_res[queue_length_frac:2*queue_length_frac], torch.unsqueeze(new_novel[0], dim=0), 
                                        queue_base_res[2*queue_length_frac:], torch.unsqueeze(new_aug[0], dim=0))) 
            queue_new_trg = torch.cat([queue_base_trg[:queue_length_frac], torch.unsqueeze(new_base[1], dim=0), 
                                        queue_base_trg[queue_length_frac:2*queue_length_frac], torch.unsqueeze(new_novel[1], dim=0), 
                                        queue_base_trg[2*queue_length_frac:], torch.unsqueeze(new_aug[1], dim=0)]) 
            queue_new_iou = torch.cat([queue_base_iou[:queue_length_frac], torch.unsqueeze(new_base[2], dim=0), 
                                        queue_base_iou[queue_length_frac:2*queue_length_frac], torch.unsqueeze(new_novel[2], dim=0), 
                                        queue_base_iou[2*queue_length_frac:], torch.unsqueeze(new_aug[2], dim=0)]) 

            self.queue_res = queue_new_res.detach()
            self.queue_trg = queue_new_trg.detach()
            self.queue_iou = queue_new_iou.detach()

        else:
            ptr = int(self.queue_ptr) 
            self.queue_res[ptr,:] = new_base[0].detach()
            self.queue_res[queue_length_frac+ptr,:] = new_novel[0].detach()
            self.queue_res[2*queue_length_frac+ptr,:] = new_aug[0].detach()

            self.queue_trg[ptr,:] = new_base[1].detach()
            self.queue_trg[queue_length_frac+ptr,:] = new_novel[1].detach()
            self.queue_trg[2*queue_length_frac+ptr,:] = new_aug[1].detach()

            self.queue_iou[ptr,:] = new_base[2].detach()
            self.queue_iou[queue_length_frac+ptr,:] = new_novel[2].detach()
            self.queue_iou[2*queue_length_frac+ptr,:] = new_aug[2].detach()

            ptr = (ptr + 1) % queue_length_frac  # move pointer

            self.queue_ptr[0] = ptr


    @torch.no_grad()
    def update_queue_2(self, new_base, new_novel, new_aug):

        
        
        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
        else:
            assert True, " One of base, novel or aug should not be none"
        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1

        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        
        ptr = int(self.queue_ptr_X) 

        if queue_base_trg is not None:
            queue_length_frac = int(queue_base_res.shape[0]/len_id)

            len_inc = 0
            if new_base is not None:
                len_inc += 1
            if new_novel is not None:
                len_inc += 1
            if new_aug is not None:
                len_inc += 1
            if queue_base_res.shape[0] <= (self.queue_length-len_inc):
                
                queue_new_res = []
                queue_new_trg = []
                queue_new_iou = []
                len_id = 0
                if new_base is not None:
                    queue_new_res += [queue_base_res[:queue_length_frac], torch.unsqueeze(new_base[0], dim=0)]
                    queue_new_trg += [queue_base_trg[:queue_length_frac], torch.unsqueeze(new_base[1], dim=0)]
                    queue_new_iou += [queue_base_iou[:queue_length_frac], torch.unsqueeze(new_base[2], dim=0)]
                    len_id += 1
                if new_novel is not None:
                    queue_new_res += [queue_base_res[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[0], dim=0).to(new_base[0].device)]
                    queue_new_trg += [queue_base_trg[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[1], dim=0).to(new_base[1].device)]
                    queue_new_iou += [queue_base_iou[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[2], dim=0).to(new_base[2].device)]
                    len_id += 1
                if new_aug is not None:
                    queue_new_res += [queue_base_res[len_id * queue_length_frac:], torch.unsqueeze(new_aug[0], dim=0).to(new_base[0].device)]
                    queue_new_trg += [queue_base_trg[len_id * queue_length_frac:], torch.unsqueeze(new_aug[1], dim=0).to(new_base[1].device)]
                    queue_new_iou += [queue_base_iou[len_id * queue_length_frac:], torch.unsqueeze(new_aug[2], dim=0).to(new_base[2].device)]

                queue_new_res = torch.cat(queue_new_res) 
                queue_new_trg = torch.cat(queue_new_trg) 
                queue_new_iou = torch.cat(queue_new_iou) 

                

                self.queue_res = queue_new_res.detach()
                self.queue_trg = queue_new_trg.detach()
                self.queue_iou = queue_new_iou.detach()

            else:
                update = True
                len_id = 0
                ptr = int(self.queue_ptr_X) 
                
                if new_base is not None:
                    if new_base[0].shape[0] != self.queue_res[ptr,:].shape[0]:
                        tmp = torch.zeros_like(self.queue_res[ptr])
                        tmp[:new_base[0].shape[0]] = new_base[0].detach()
                        self.queue_res[ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_trg[ptr])
                        tmp[:new_base[1].shape[0]] = new_base[1].detach()
                        self.queue_trg[ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_iou[ptr])
                        tmp[:new_base[2].shape[0]] = new_base[2].detach()
                        self.queue_iou[ptr,:] = tmp
                        del tmp
                    else:
                        self.queue_res[ptr,:] = new_base[0].detach()
                        self.queue_trg[ptr,:] = new_base[1].detach()
                        self.queue_iou[ptr,:] = new_base[2].detach()
                    len_id += 1
                queue_shape = self.queue_trg[len_id * queue_length_frac+ptr,:].shape[0]

                if new_novel is not None:
                    if (new_novel[0].shape[0] != queue_shape) or (new_novel[1].shape[0] != queue_shape) or (new_novel[2].shape[0] != queue_shape):
                        tmp = torch.zeros_like(self.queue_res[ptr])
                        tmp[:new_novel[0].shape[0]] = new_novel[0].detach()
                        
                        self.queue_res[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_trg[ptr])
                        tmp[:new_novel[1].shape[0]] = new_novel[1].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_iou[ptr])
                        tmp[:new_novel[2].shape[0]] = new_novel[2].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = tmp
                        len_id += 1
                        del tmp
                        #update = False
                    else:
                        self.queue_res[len_id * queue_length_frac+ptr,:] = new_novel[0].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = new_novel[1].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = new_novel[2].detach()
                        len_id += 1
                if new_aug is not None:
                    if (new_aug[0].shape[0] != queue_shape) or (new_aug[1].shape[0] != queue_shape) or (new_aug[2].shape[0] != queue_shape):
                        tmp = torch.zeros_like(self.queue_res[ptr])
                        tmp[:new_aug[0].shape[0]] = new_aug[0].detach()
                        
                        self.queue_res[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_trg[ptr])
                        tmp[:new_aug[1].shape[0]] = new_aug[1].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_iou[ptr])
                        tmp[:new_aug[2].shape[0]] = new_aug[2].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = tmp
                        len_id += 1
                        del tmp
                        #update = False
                    else:
                        self.queue_res[len_id * queue_length_frac+ptr,:] = new_aug[0].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = new_aug[1].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = new_aug[2].detach()
                        len_id += 1
                
                if update:

                    ptr = (ptr + len_id) % queue_length_frac
                    #ptr = (ptr + 1) % queue_length_frac  # move pointer OLD & WRONG

                    self.queue_ptr[0] = ptr
                    self.queue_ptr_X[0] = ptr
        else:                
            queue_new_res = []
            queue_new_trg = []
            queue_new_iou = []
            len_id = 0
            if new_base is not None:
                queue_new_res += [torch.unsqueeze(new_base[0], dim=0)]
                queue_new_trg += [torch.unsqueeze(new_base[1], dim=0)]
                queue_new_iou += [torch.unsqueeze(new_base[2], dim=0)]
                len_id += 1
            if new_novel is not None:
                queue_new_res += [torch.unsqueeze(new_novel[0], dim=0).to(new_base[0].device)]
                queue_new_trg += [torch.unsqueeze(new_novel[1], dim=0).to(new_base[1].device)]
                queue_new_iou += [torch.unsqueeze(new_novel[2], dim=0).to(new_base[2].device)]
                len_id += 1
            if new_aug is not None:
                queue_new_res += [torch.unsqueeze(new_aug[0], dim=0).to(new_base[0].device)]
                queue_new_trg += [torch.unsqueeze(new_aug[1], dim=0).to(new_base[1].device)]
                queue_new_iou += [torch.unsqueeze(new_aug[2], dim=0).to(new_base[2].device)]

            queue_new_res = torch.cat(queue_new_res) 
            queue_new_trg = torch.cat(queue_new_trg) 
            queue_new_iou = torch.cat(queue_new_iou) 
            
            self.queue_res = queue_new_res.detach()
            self.queue_trg = queue_new_trg.detach()
            self.queue_iou = queue_new_iou.detach()
        if self.save_bbox_feat:
            self.save_queue(new_base, new_novel, new_aug)
    
    def save_queue(self,new_base, new_novel, new_aug):
        reshaped_trg = self.queue_trg.clone().reshape(-1)
        reshaped_iou = self.queue_iou.clone().reshape(-1)
        reshaped_res = self.queue_res.clone().reshape(-1, self.queue_res.shape[2])
        queue_dict = {}
        for class_id in range(self.num_classes+1):
            if class_id in reshaped_trg.unique():
                msk = reshaped_trg == class_id
                res_msk = reshaped_res[msk]
                iou_msk = reshaped_iou[msk]
                trg_msk = reshaped_trg[msk]
                tmp_dict ={'results':res_msk, 'trg':trg_msk, 'iou':iou_msk} 
                queue_dict[class_id]=tmp_dict
        with open(self.queue_path, 'wb') as fp:
            pickle.dump(queue_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            self.save_bbox_feat = False   

    def load_queue(self, base=True, novel=True, aug=True):
        with open(self.queue_path, 'rb') as fp:
            data = pickle.load(fp)
        queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg = [], [], [], [], [], []
        queue_base_iou, queue_novel_iou, queue_aug_iou = [], [], []
        for key in data.keys():
            queue_base_res.append(data[key]['base_results'])
            queue_base_trg.append(data[key]['base_trg'])
            queue_novel_res.append(data[key]['novel_results'])
            queue_novel_trg.append(data[key]['novel_trg'])
            queue_aug_res.append(data[key]['aug_results'])
            queue_aug_trg.append(data[key]['aug_trg'])
            queue_base_iou.append(data[key]['base_ious'])
            queue_novel_iou.append(data[key]['novel_ious'])
            queue_aug_iou.append(data[key]['aug_ious'])
        queue_base_res = queue_base_res + queue_novel_res + queue_aug_res
        queue_base_res = torch.stack(queue_base_res)

        queue_base_trg = queue_base_trg + queue_novel_trg + queue_aug_trg
        queue_base_trg = torch.stack(queue_base_trg)

        queue_base_iou = queue_base_iou + queue_novel_iou + queue_aug_iou
        queue_base_iou = torch.stack(queue_base_iou)

        id_shuffle = np.arange(queue_base_res.shape[0])
        np.random.shuffle(id_shuffle)
        queue_base_res = queue_base_res[id_shuffle]
        queue_base_trg = queue_base_trg[id_shuffle]
        queue_base_iou = queue_base_iou[id_shuffle]
        
        return queue_base_res, queue_base_trg, queue_base_iou

    def load_queue_2(self, base=True, novel=True, aug=True):
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou = [], [], []
            for key in data.keys():
                if base:
                    queue_res.append(data[key]['base_results'])
                    queue_trg.append(data[key]['base_trg'])
                    queue_iou.append(data[key]['base_ious'])

                if novel:  
                    queue_res.append(data[key]['novel_results'])
                    queue_trg.append(data[key]['novel_trg'])
                    queue_iou.append(data[key]['novel_ious'])

                if aug:
                    queue_res.append(data[key]['aug_results'])
                    queue_trg.append(data[key]['aug_trg'])
                    queue_iou.append(data[key]['aug_ious'])


            queue_base_res = torch.stack(queue_res)
            queue_base_trg = torch.stack(queue_trg)
            queue_base_iou = torch.stack(queue_iou)


            id_shuffle = np.arange(queue_base_res.shape[0])
            np.random.shuffle(id_shuffle)
            queue_base_res = queue_base_res[id_shuffle]
            queue_base_trg = queue_base_trg[id_shuffle]
            queue_base_iou = queue_base_iou[id_shuffle]
        else:
            queue_base_res, queue_base_trg, queue_base_iou = torch.zeros(1), None, torch.zeros(1)

        return queue_base_res, queue_base_trg, queue_base_iou
    
    def load_queue_3(self, base=True, novel=True, aug=True):
        if self.queue_path is not None:
            if exists(self.queue_path):
                with open(self.queue_path, 'rb') as fp:
                    data = pickle.load(fp)
                queue_res, queue_trg, queue_iou = [], [], []
                for key in data.keys():
                    for second_key in data[key].keys():
                        if 'results' in second_key:
                            queue_res.append(data[key][second_key])
                        if 'trg' in second_key:
                            queue_trg.append(data[key][second_key])
                        if 'iou' in second_key:
                            queue_iou.append(data[key][second_key])
                queue_base_res = torch.vstack(queue_res).reshape(-1, self.num_proposals, 128)
                queue_base_trg = torch.hstack(queue_trg).reshape(-1, self.num_proposals)
                queue_base_iou = torch.hstack(queue_iou).reshape(-1, self.num_proposals)
                
                id_shuffle = np.arange(queue_base_res.shape[0])
                np.random.shuffle(id_shuffle)
                queue_base_res = queue_base_res[id_shuffle]
                queue_base_trg = queue_base_trg[id_shuffle]
                queue_base_iou = queue_base_iou[id_shuffle]
            else:
                queue_base_res, queue_base_trg, queue_base_iou = torch.zeros(1), None, torch.zeros(1)
        else:
            queue_base_res, queue_base_trg, queue_base_iou = torch.zeros(1), None, torch.zeros(1)

        return queue_base_res, queue_base_trg, queue_base_iou

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
            cls_score, bbox_pred, contrast_feat  = self.forward_one_branch(x, True)
            aug_cls_score, aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug, True)
        else:
            cls_score, bbox_pred, aug_cls_score, aug_bbox_pred = None, None, None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_cls_score, base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_cls_score, base_bbox_pred, base_contrast_feat, cls_score, bbox_pred, contrast_feat, aug_cls_score, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x, base=True):
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

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if self.to_norm:
        # cls branch
            if x_cls.dim() > 2:
                x_cls = torch.flatten(x_cls, start_dim=1)

            
            # normalize the input x along the `input_size` dimension
            x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_cls_normalized = x_cls.div(x_norm + self.eps).clone()
            # normalize weight
            with torch.no_grad():
                temp_norm = torch.norm(self.fc_cls.weight, p=2,dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
                self.fc_cls.weight.clone().div_(temp_norm + self.eps)
            # calculate and scale cls_score
            
            if base: 
                cls_output = self.fc_cls(x_cls_normalized)
                cls_score = self.scale * cls_output if self.with_cls else None
            else:
                with torch.no_grad():
                    cls_output = self.fc_cls(x_cls_normalized)
                    cls_score = self.scale * cls_output if self.with_cls else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None

        # contrastive branch
        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)

        return cls_score, bbox_pred, cont_feat




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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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
        if self.main_training and not self.same_class:
            if self.with_weight_decay:
                decay_rate = self._decay_rate
            if self.contrast_loss_base_aug is not None:

                losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)   
            if self.contrast_loss_cls is not None:
                losses['loss_c_cls'] = self.contrast_loss_cls(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss is not None:
                losses['loss_cosine'] = self.contrast_loss(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss_all is not None:
                losses['loss_contrastive'] = self.contrast_loss_all(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    queue_res, 
                    queue_trg, 
                    queue_iou,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
        elif self.main_training and self.same_class_all and self.use_queue:
            classes_eq[num_classes]=num_classes
            if self.use_base_queue:
                if (gt_base_bbox[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                else:
                    queue_trg = gt_base_bbox[0]
                base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
            else:
                base_update = None
            if self.use_novel_queue:
                if (bbox_targets[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                else:
                    queue_trg = bbox_targets[0]
                novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
            else:
                novel_update = None

            if self.use_aug_queue:
                if (aug_bbox_targets[0] > num_classes).any():
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                else:
                    queue_trg = aug_bbox_targets[0]
                aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
            else:
                aug_update = None

            self.update_queue_2(base_update, novel_update, aug_update )
            if self.queue_trg is None:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            else:
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            
            self.id_save_FT += 1   
        elif self.main_training and self.same_class and self.use_queue:
            if self.queue_trg is None:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            else:
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            classes_eq[num_classes]=num_classes
            if self.use_base_queue:
                if (gt_base_bbox[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                else:
                    queue_trg = gt_base_bbox[0]
                base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
            else:
                base_update = None
            if self.use_novel_queue:
                if (bbox_targets[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                else:
                    queue_trg = bbox_targets[0]
                novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
            else:
                novel_update = None

            if self.use_aug_queue:
                if (aug_bbox_targets[0] > num_classes).any():
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                else:
                    queue_trg = aug_bbox_targets[0]
                aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
            else:
                aug_update = None

            self.update_queue_2(base_update, novel_update, aug_update )
            self.id_save_FT += 1   
        
        elif self.main_training and self.same_class:
            if self.with_weight_decay:
                decay_rate = self._decay_rate
            if self.contrast_loss_base_aug is not None:

                losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)   
            if self.contrast_loss_cls is not None:
                losses['loss_c_cls'] = self.contrast_loss_cls(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss is not None:
                losses['loss_cosine'] = self.contrast_loss(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss_all is not None:
                losses['loss_contrastive'] = self.contrast_loss_all(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
        elif self.queue_res is not None and self.use_queue:
            #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
            
            # base novel aug
            if self.queue_trg is None:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cls_score'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cls_score'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cls_score'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            else:
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            
            classes_eq[num_classes]=num_classes
            if self.use_base_queue:
                if (gt_base_bbox[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                else:
                    queue_trg = gt_base_bbox[0]
                base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
            else:
                base_update = None
            if self.use_novel_queue:
                if (bbox_targets[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                else:
                    queue_trg = bbox_targets[0]
                novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
            else:
                novel_update = None

            if self.use_aug_queue:
                if (aug_bbox_targets[0] > num_classes).any():
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                else:
                    queue_trg = aug_bbox_targets[0]
                aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
            else:
                aug_update = None

            self.update_queue_2(base_update, novel_update, aug_update )
            self.id_save_FT += 1   
        
        else:

            if base_proposal_ious is not None:

                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            else:

                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        bbox_results['cont_feat'], 
                        gt_base_bbox[0], 
                        bbox_targets[0],
                        classes_eq, 
                        num_classes,
                        reduction_override=reduction_override)
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        aug_bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        aug_bbox_targets[0], 
                        None, 
                        num_classes,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
    
        if self.contrast_loss_bbox is not None and False:
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
        
        if self.contrast_loss_bbox is not None:
            pos_inds_aug = (aug_bbox_targets[0] >= 0) & (aug_bbox_targets[0] > self.num_classes) 
            pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] > self.num_classes)
            base_pos_inds = (gt_base_bbox[0] >= 0) & (gt_base_bbox[0] < self.num_classes)
            
            # do not perform bounding box regression for BG anymore.
            
            if pos_inds.any() and pos_inds_aug.any():
                # novel
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_score_pred.view(
                        bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
                else:
                    bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                    labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                    pos_bbox_pred = bbox_score_pred.view(
                        bbox_score_pred.shape[0], -1, 4)[pos_inds.type(torch.bool),labels_true]
                # aug
                if self.reg_class_agnostic:
                    aug_pos_bbox_pred = aug_bbox_score_pred.view(
                        aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
                else:
                    bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                    labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                    aug_pos_bbox_pred = aug_bbox_score_pred.view(
                        aug_bbox_score_pred.shape[0], -1,
                        4)[pos_inds_aug.type(torch.bool),labels_true_aug]
                # base
                if self.reg_class_agnostic:
                    base_pos_bbox_pred = base_bbox_pred['bbox_pred'].view(
                        base_bbox_pred['bbox_pred'].shape[0], 4)[base_pos_inds.type(torch.bool)]
                else:
                    bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                    base_pos_bbox_pred = base_bbox_pred['bbox_pred'].view(
                        base_bbox_pred['bbox_pred'].shape[0], -1,
                        4)[base_pos_inds.type(torch.bool),bbox_label_base]
                
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
                base_pos_bbox_pred, 
                bbox_label_base,
                classes_eq,
                transform_applied,
                gt_labels_aug,
                min_size,
                base_bbox_pred,
                gt_base_bbox,
                reduction_override=reduction_override)
        
        if not self.use_queue and not self.main_training:
            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)
            if (aug_bbox_targets[0]> num_classes).any():
                aug_bbox_save = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0]]).to(aug_bbox_results['cont_feat'].device)

            cont_dict = {
                'base_results': base_bbox_pred['cont_feat'], 
                'base_trg': gt_base_bbox[0], 
                'base_ious': base_proposal_ious,
                'novel_results': bbox_results['cont_feat'], 
                'novel_trg': bbox_save, 
                'novel_ious': proposal_ious,
                'aug_results': aug_bbox_results['cont_feat'], 
                'aug_trg': aug_bbox_save,
                'aug_ious': aug_proposal_ious
                }
            id_dict = {str(self.id_save_FT):cont_dict}
            self.id_save_FT += 1
            
            self.queue_init(id_dict)
        if 'loss_cosine' in losses.keys():
            if losses['loss_cosine'] == -1:
                print(f'no loss')
                losses = dict()
        return losses

@HEADS.register_module()
class QueueAugContrastiveBBoxHead_Branch_classqueue(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 loss_all = None,
                 contrast_loss_classif = None,
                 to_norm_cls = False,
                 main_training = False,
                 same_class = False,
                 same_class_all = False,
                 queue_path = 'init_queue.p',
                 init_queue = False,
                 use_queue = False,
                 use_base_queue = True,
                 use_novel_queue = True,
                 use_aug_queue = True,
                 queue_length = 60,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_all is None:
            if loss_c_cls is not None:
                self.contrast_loss_cls = build_loss(loss_c_cls)
            else:
                self.contrast_loss_cls = None

            if loss_cosine is not None:
                self.contrast_loss = build_loss(loss_cosine)
            else:
                self.contrast_loss = None
            if loss_base_aug is not None:
                self.contrast_loss_base_aug = build_loss(loss_base_aug)
            else:
                self.contrast_loss_base_aug = None
            if contrast_loss_classif is not None:
                self.contrast_loss_classif = build_loss(contrast_loss_classif)
            else:
                self.contrast_loss_classif = None
            self.contrast_loss_all = None
        else:
            self.contrast_loss_all = build_loss(loss_all)
            self.contrast_loss_cls = None
            self.contrast_loss = None
            self.contrast_loss_base_aug = None


        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.queue_length = queue_length
        if self.use_queue and not init_queue:
            queue_base_res, queue_base_trg, queue_base_iou = self.load_queue_class(use_base_queue, use_novel_queue, use_aug_queue)

            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        else:
            self.queue_res = None

        self.to_norm = to_norm_cls


        self.same_class_all = same_class_all
        if self.same_class_all:
            self.same_class = False
        else: 
            self.same_class = same_class
        self.main_training = main_training
        
        self.id_save_FT = 1
        self.init_queue = init_queue
        if init_queue:
            self.queue_new_res = torch.empty((self.num_classes, mlp_head_channels))
            self.queue_new_trg = -torch.ones((self.num_classes))
            self.queue_new_iou = torch.ones((self.num_classes))

    def queue_init(self, new_dict):
        if self.queue_res is None:
            self.queue_res = new_dict
        else:
            z = self.queue_res.copy()   
            z.update(new_dict)
            self.queue_res = z

        with open(self.queue_path, 'wb') as fp:
            pickle.dump(self.queue_res, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @torch.no_grad()
    def update_queue_class(self, new_base, new_novel, new_aug):

        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
            fs = new_base[0].shape[1]
            cur_device = new_base[0].device
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
            fs = new_novel[0].shape[1]
            cur_device = new_novel[0].device
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
            fs = new_aug[0].shape[1]
            cur_device = new_aug[0].device
        else:
            assert True, " One of base, novel or aug should not be none"
        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1
        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        if queue_base_trg is not None:
            update = True
            ptr = int(self.queue_ptr) 
            len_id = 0
            for class_id in range(self.num_classes):
                tmp = None
                if new_base is not None:
                    if class_id in new_base[1].unique():
                        if queue_base_trg[class_id] == -1:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                        else:
                            tmp = torch.vstack([queue_base_res[class_id], new_base[0][new_base[1][new_base[1] == class_id]]])

                if new_novel is not None:
                    if class_id in new_novel[1].unique():
                        if tmp is not None:
                            tmp = torch.vstack([tmp, new_novel[0][new_novel[1][new_novel[1] == class_id]]])
                        elif queue_base_trg[class_id] == -1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                        else:
                            tmp = torch.vstack([queue_base_res[class_id], new_novel[0][new_novel[1][new_novel[1] == class_id]]])


                if new_aug is not None:
                    if class_id in new_aug[1].unique():
                        if tmp is not None:
                            tmp = torch.vstack([tmp, new_aug[0][new_aug[1][new_aug[1] == class_id]]])
                        elif queue_base_trg[class_id] == -1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                        else:
                            tmp = torch.vstack([queue_base_res[class_id], new_aug[0][new_aug[1][new_aug[1] == class_id]]])


                if tmp is not None:
                    queue_base_res[class_id] = tmp.mean(dim=0)
                    queue_base_trg[class_id] = class_id


            self.queue_res = queue_base_res.to(cur_device).detach()
            self.queue_trg = queue_base_trg.to(cur_device).detach()
            self.queue_iou = queue_base_iou.to(cur_device).detach()
        else:                
            queue_new_res = torch.empty((self.num_classes, fs))
            queue_new_trg = -torch.ones((self.num_classes))
            queue_new_iou = torch.ones((self.num_classes))
            len_id = 0
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_novel is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_novel[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_novel[0][new_novel[1][new_novel[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_aug is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_aug[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_aug[0][new_aug[1][new_aug[1] == class_id]].mean(dim=0)


            
            self.queue_res = queue_new_res.to(cur_device).detach()
            self.queue_trg = queue_new_trg.to(cur_device).detach()
            self.queue_iou = queue_new_iou.to(cur_device).detach()
    
    def load_queue_class(self, base=True, novel=True, aug=True):
        
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou = [], [], []
            for key_class in range(self.num_classes):
                queue_res.append(data[str(key_class)])
                queue_trg.append(key_class)
                queue_iou.append(1.)
            queue_base_res = torch.stack(queue_res)
            queue_base_trg = torch.tensor(queue_trg)
            queue_base_iou = torch.tensor(queue_iou)
        else:
            queue_base_res, queue_base_trg, queue_base_iou = torch.zeros(1), None, torch.zeros(1)
        return queue_base_res, queue_base_trg, queue_base_iou

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
            cls_score, bbox_pred, contrast_feat  = self.forward_one_branch(x, True)
            aug_cls_score, aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug, True)
        else:
            cls_score, bbox_pred, aug_cls_score, aug_bbox_pred = None, None, None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_cls_score, base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_cls_score, base_bbox_pred, base_contrast_feat, cls_score, bbox_pred, contrast_feat, aug_cls_score, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x, base=True):
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

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if self.to_norm:
        # cls branch
            if x_cls.dim() > 2:
                x_cls = torch.flatten(x_cls, start_dim=1)

            
            # normalize the input x along the `input_size` dimension
            x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_cls_normalized = x_cls.div(x_norm + self.eps).clone()
            # normalize weight
            with torch.no_grad():
                temp_norm = torch.norm(self.fc_cls.weight, p=2,dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
                self.fc_cls.weight.clone().div_(temp_norm + self.eps)
            # calculate and scale cls_score
            
            if base: 
                cls_output = self.fc_cls(x_cls_normalized)
                cls_score = self.scale * cls_output if self.with_cls else None
            else:
                with torch.no_grad():
                    cls_output = self.fc_cls(x_cls_normalized)
                    cls_score = self.scale * cls_output if self.with_cls else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None

        # contrastive branch

        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)
        return cls_score, bbox_pred, cont_feat


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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}

            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
            losses = dict()
        else:
            if self.main_training and not self.same_class and not self.use_queue:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            elif self.main_training and self.same_class and self.use_queue:
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    classes_eq[num_classes]=num_classes
                    if self.use_base_queue:
                        if (gt_base_bbox[0] > num_classes).any():                    
                            queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                        else:
                            queue_trg = gt_base_bbox[0]
                        base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                    else:
                        base_update = None
                    if self.use_novel_queue:
                        if (bbox_targets[0] > num_classes).any():                    
                            queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                        else:
                            queue_trg = bbox_targets[0]
                        novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                    else:
                        novel_update = None

                    if self.use_aug_queue:
                        if (aug_bbox_targets[0] > num_classes).any():
                            queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                        else:
                            queue_trg = aug_bbox_targets[0]
                        aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                    else:
                        aug_update = None

                    self.update_queue_class(base_update, novel_update, aug_update )
                
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                self.id_save_FT += 1   
            elif self.main_training and self.same_class_all and self.use_queue:
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                

                self.id_save_FT += 1   
            
            elif self.main_training and self.same_class:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override) 
            elif self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                
                # base novel aug
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cls_score'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cls_score'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cls_score'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
            else:

                if base_proposal_ious is not None:

                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                else:

                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            bbox_results['cont_feat'], 
                            gt_base_bbox[0], 
                            bbox_targets[0],
                            classes_eq, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            aug_bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            aug_bbox_targets[0], 
                            None, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
        
            if self.contrast_loss_bbox is not None:
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
            
            
            if 'loss_cosine' in losses.keys():
                if losses['loss_cosine'] == -1:
                    print(f'no loss')
                    losses = dict()
        return losses
   
    @force_fp32(apply_to=('cont_feat'))
    def loss_contrast_cls(self,
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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
        else:
            if self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                
                losses['loss_contrastive_cls'], pred_cont = self.contrast_loss_classif(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    queue_res, 
                    queue_trg, 
                    queue_iou,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    reduction_override=reduction_override)
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
        
        return losses
   
@HEADS.register_module()
class QueueAugContrastiveBBoxHead_Branch_classqueue_replace(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 loss_all = None,
                 contrast_loss_classif = None,
                 to_norm_cls = False,
                 main_training = False,
                 same_class = False,
                 same_class_all = False,
                 queue_path = 'init_queue.p',
                 init_queue = False,
                 use_queue = False,
                 use_base_queue = True,
                 use_novel_queue = True,
                 use_aug_queue = True,
                 queue_length = 60,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_all is None:
            if loss_c_cls is not None:
                self.contrast_loss_cls = build_loss(loss_c_cls)
            else:
                self.contrast_loss_cls = None

            if loss_cosine is not None:
                self.contrast_loss = build_loss(loss_cosine)
            else:
                self.contrast_loss = None
            if loss_base_aug is not None:
                self.contrast_loss_base_aug = build_loss(loss_base_aug)
            else:
                self.contrast_loss_base_aug = None
            if contrast_loss_classif is not None:
                self.contrast_loss_classif = build_loss(contrast_loss_classif)
            else:
                self.contrast_loss_classif = None
            self.contrast_loss_all = None
        else:
            self.contrast_loss_all = build_loss(loss_all)
            self.contrast_loss_cls = None
            self.contrast_loss = None
            self.contrast_loss_base_aug = None


        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.queue_length = queue_length
        if self.use_queue and not init_queue:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = self.load_queue_class(use_base_queue, use_novel_queue, use_aug_queue)
            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', queue_base_ptr)
        else:
            self.queue_res = None

        self.to_norm = to_norm_cls


        self.same_class_all = same_class_all
        if self.same_class_all:
            self.same_class = False
        else: 
            self.same_class = same_class
        self.main_training = main_training
        
        self.id_save_FT = 1
        self.init_queue = init_queue
        if init_queue:
            self.queue_new_res = torch.empty((self.queue_length, mlp_head_channels))
            self.queue_new_trg = -torch.ones((self.queue_length))
            self.queue_new_iou = torch.ones((self.queue_length))


    def queue_init(self, new_dict):
        if self.queue_res is None:
            self.queue_res = new_dict
        else:
            z = self.queue_res.copy()   
            z.update(new_dict)
            self.queue_res = z

        with open(self.queue_path, 'wb') as fp:
            pickle.dump(self.queue_res, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @torch.no_grad()
    def update_queue_class(self, new_base, new_novel, new_aug):
        
        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
            fs = new_base[0].shape[1]
            cur_device = new_base[0].device
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
            fs = new_novel[0].shape[1]
            cur_device = new_novel[0].device
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
            fs = new_aug[0].shape[1]
            cur_device = new_aug[0].device
        else:
            assert True, " One of base, novel or aug should not be none"

        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1
        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        if queue_base_trg is not None:
            update = True
            ptr = self.queue_ptr.int() 
            reset = False
            increase = False
            for class_id in range(self.num_classes):
                tmp = None
                if new_base is not None:
                    if class_id in new_base[1].unique():
                        if ptr[class_id] == (self.queue_length//self.num_classes)-1:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                            
                if new_novel is not None:
                    if class_id in new_novel[1].unique():
                        if tmp is not None:
                            tmp2 = new_novel[0][new_novel[1][new_novel[1] == class_id]]

                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes)-1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True



                if new_aug is not None:
                    if class_id in new_aug[1].unique():
                        if tmp is not None:
                            tmp2 = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes)-1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True

                if tmp is not None:
                    tmp = tmp.mean(dim=0)
                    queue_base_res[class_id+(self.num_classes*ptr[class_id])] = tmp
                    queue_base_trg[class_id+(self.num_classes*ptr[class_id])] = class_id

                    if reset and increase:
                        assert True, "problem, reset and increase"
                    elif reset:
                        ptr[class_id] = 0
                    elif increase:
                        ptr[class_id] += 1
            self.queue_res = queue_base_res.to(cur_device).detach()
            self.queue_trg = queue_base_trg.to(cur_device).detach()
            self.queue_ptr = ptr
            self.queue_iou = queue_base_iou.to(cur_device).detach()
            
        else:                
            shape_queue = (self.queue_length//self.num_classes)*self.num_classes
            queue_new_res = torch.empty((shape_queue, fs))
            queue_new_trg = -torch.ones((shape_queue))
            queue_new_iou = torch.ones((shape_queue))
            len_id = 0
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_novel is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_novel[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_novel[0][new_novel[1][new_novel[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_aug is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_aug[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_aug[0][new_aug[1][new_aug[1] == class_id]].mean(dim=0)


            
            self.queue_res = queue_new_res.to(cur_device).detach()
            self.queue_trg = queue_new_trg.to(cur_device).detach()
            self.queue_iou = queue_new_iou.to(cur_device).detach()
    
    def load_queue_class_new(self, base=True, novel=True, aug=True):
        
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
            bg_base = -1
            
            
            if 'bg' in data.keys():
                bg_base = data['bg']['trg'].item()
            for k in data.keys():
                key_type = type(k)
                break
            
            
            for key_class in range(len(data.keys())-1):#self.num_classes):
                if bg_base == key_class:
                    queue_res.append(data['bg'])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                else:
                    if key_type is str:
                        if type(data[str(key_class)]) is Dict:
                            queue_res.append(data[str(key_class)]['results'])
                        else:
                            queue_res.append(data[str(key_class)])
                    else:
                        if type(data[key_class]) is Dict:
                            queue_res.append(data[key_class]['results'])
                        else:
                            queue_res.append(data[key_class])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                queue_ptr.append(1.)
            queue_new_ptr = torch.tensor(queue_ptr).long()
            if (len(data.keys())-1) < self.num_classes:
                queue_base_ptr = torch.zeros(self.num_classes, dtype=torch.long)
                queue_base_ptr[:queue_new_ptr.shape[0]] = queue_new_ptr.clone()
            else:
                queue_base_ptr = queue_new_ptr.clone()

            queue_new_res = torch.stack(queue_res)
            queue_new_trg = torch.tensor(queue_trg)
            queue_new_iou = torch.tensor(queue_iou)

            fs = queue_new_res.shape[1]
            shape_queue = (self.queue_length//self.num_classes)*self.num_classes

            queue_base_res = torch.zeros((shape_queue, fs))
            queue_base_res[:queue_new_res.shape[0]] = queue_new_res.clone()

            queue_base_trg = -torch.ones((shape_queue))
            queue_base_trg[:queue_new_trg.shape[0]] = queue_new_trg.clone()

            queue_base_iou = torch.ones((shape_queue))
            queue_base_iou[:queue_new_iou.shape[0]] = queue_new_iou.clone()

        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

    def load_queue_class_bu(self, base=True, novel=True, aug=True):
        
        if self.queue_path is not None:
            if exists(self.queue_path):
                with open(self.queue_path, 'rb') as fp:
                    data = pickle.load(fp)
                queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
                
                for key_class in range(self.num_classes):
                    queue_res.append(data[str(key_class)])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                    queue_ptr.append(1.)
                queue_new_ptr = torch.tensor(queue_ptr).long()
                queue_base_ptr = queue_new_ptr.clone()

                queue_new_res = torch.stack(queue_res)
                queue_new_trg = torch.tensor(queue_trg)
                queue_new_iou = torch.tensor(queue_iou)

                fs = queue_new_res.shape[1]
                shape_queue = (self.queue_length//self.num_classes)*self.num_classes

                queue_base_res = torch.zeros((shape_queue, fs))
                queue_base_res[:queue_new_res.shape[0]] = queue_new_res.clone()

                queue_base_trg = -torch.ones((shape_queue))
                queue_base_trg[:queue_new_trg.shape[0]] = queue_new_trg.clone()

                queue_base_iou = torch.ones((shape_queue))
                queue_base_iou[:queue_new_iou.shape[0]] = queue_new_iou.clone()

        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

    def load_queue_class(self, base=True, novel=True, aug=True):
        if self.queue_path is not None:
            if exists(self.queue_path):
                with open(self.queue_path, 'rb') as fp:
                    data = pickle.load(fp)
                queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
                nbr_rep_in_queue = self.queue_length//(self.num_classes)
                for key_class in range(self.num_classes):
                    if str(key_class) not in data.keys():
                        queue_res.append(torch.zeros(128))
                        queue_trg.append(-1)
                        queue_iou.append(-1.)
                        queue_ptr.append(0.)
                    else:
                        if len(data[str(key_class)].shape) > 1:
                            tmp_res, tmp_trg, tmp_iou = [], [], []
                            tmp_ptr = 0
                            for rep_id in range(data[str(key_class)].shape[0]):
                                tmp_res.append(data[str(key_class)][rep_id])
                                tmp_trg.append(key_class)
                                tmp_iou.append(1.)
                                tmp_ptr += 1.
                            if tmp_ptr >= nbr_rep_in_queue:
                                tmp_ptr -= 1
                            queue_res.append(tmp_res)
                            queue_trg.append(tmp_trg)
                            queue_iou.append(tmp_iou)
                            queue_ptr.append(tmp_ptr)
                            del tmp_res, tmp_trg, tmp_iou
                        else:
                            queue_res.append(data[str(key_class)])
                            queue_trg.append(key_class)
                            queue_iou.append(1.)
                            queue_ptr.append(1.)
                queue_new_ptr = torch.tensor(queue_ptr).long()
                queue_base_ptr = queue_new_ptr.clone()

                fs = 128
                shape_queue = (self.queue_length//(self.num_classes))*(self.num_classes)
                
                queue_base_res = torch.zeros((shape_queue, fs))
                queue_base_trg = -torch.ones((shape_queue))
                queue_base_iou = torch.ones((shape_queue))

                for rep_id in range(queue_base_ptr.shape[0]):
                    if queue_base_ptr[rep_id] > 1.:
                        
                        for nbr_rep_id in range(len(queue_res[rep_id])):
                            if nbr_rep_id < nbr_rep_in_queue:
                                if type(queue_trg[rep_id][nbr_rep_id]) is int:
                                    rnd_id = [i for i in range(1)]
                                else:
                                    rnd_id = [i for i in range(queue_trg[rep_id][nbr_rep_id].shape[0])]
                                    rnd_id = np.random.choices(rnd_id, shape_queue)
                                queue_base_res[rep_id+nbr_rep_id*(self.num_classes)] = queue_res[rep_id][nbr_rep_id]
                                queue_base_trg[rep_id+nbr_rep_id*(self.num_classes)] = queue_trg[rep_id][nbr_rep_id]
                                queue_base_iou[rep_id+nbr_rep_id*(self.num_classes)] = queue_iou[rep_id][nbr_rep_id]
                    else:
                        queue_new_res = queue_res[rep_id]
                        queue_new_trg = queue_trg[rep_id]
                        queue_new_iou = queue_iou[rep_id]

                        queue_base_res[rep_id] = queue_new_res#.clone()
                        queue_base_trg[rep_id] = queue_new_trg#.clone()
                        queue_base_iou[rep_id] = queue_new_iou#.clone()
            else:
                queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes, dtype=torch.long)
        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

    

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
            cls_score, bbox_pred, contrast_feat  = self.forward_one_branch(x, True)
            aug_cls_score, aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug, True)
        else:
            cls_score, bbox_pred, aug_cls_score, aug_bbox_pred = None, None, None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_cls_score, base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_cls_score, base_bbox_pred, base_contrast_feat, cls_score, bbox_pred, contrast_feat, aug_cls_score, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x, base=True):
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

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # contrastive branch

        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)

        
        if self.to_norm:
        # cls branch
            if x_cls.dim() > 2:
                x_cls = torch.flatten(x_cls, start_dim=1)

            # normalize the input x along the `input_size` dimension
            x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_cls_normalized = x_cls.div(x_norm + self.eps).clone()
            # normalize weight
            with torch.no_grad():
                temp_norm = torch.norm(self.fc_cls.weight, p=2,dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
                self.fc_cls.weight.clone().div_(temp_norm + self.eps)
            # calculate and scale cls_score
            
            if base: 
                cls_output = self.fc_cls(x_cls_normalized)
                cls_score = self.scale * cls_output if self.with_cls else None
            else:
                with torch.no_grad():
                    cls_output = self.fc_cls(x_cls_normalized)
                    cls_score = self.scale * cls_output if self.with_cls else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None

        
        return cls_score, bbox_pred, cont_feat


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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}

            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
            losses = dict()
        else:
            if self.main_training and not self.same_class and not self.use_queue:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            elif self.main_training and self.same_class and self.use_queue:
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1   
            elif self.main_training and self.same_class_all and self.use_queue:
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                
                self.id_save_FT += 1   
            
            elif self.main_training and self.same_class:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override) 
            elif self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                # base novel aug
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                
            
                self.id_save_FT += 1          
            else:

                if base_proposal_ious is not None:

                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                else:

                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            bbox_results['cont_feat'], 
                            gt_base_bbox[0], 
                            bbox_targets[0],
                            classes_eq, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            aug_bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            aug_bbox_targets[0], 
                            None, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
        
            if self.contrast_loss_bbox is not None:
                pos_inds_aug = (aug_bbox_targets[0] >= 0) & (aug_bbox_targets[0] > self.num_classes) 
                pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] > self.num_classes)
                base_pos_inds = (gt_base_bbox[0] >= 0) & (gt_base_bbox[0] < self.num_classes)
                
                
                # do not perform bounding box regression for BG anymore.
                
                if pos_inds.any() and pos_inds_aug.any():
                    # novel
                    if self.reg_class_agnostic:
                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
                    else:
                        bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                        labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], -1, 4)
                        pos_bbox_pred = pos_bbox_pred[pos_inds.type(torch.bool),labels_true]

                    # aug
                    if self.reg_class_agnostic:
                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
                    else:
                        bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                        labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], -1,
                            4)[pos_inds_aug.type(torch.bool),labels_true_aug]
                    # base
                    if self.reg_class_agnostic:
                        base_pos_bbox_pred = gt_base_bbox[2].view(
                            gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                    else:
                        bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                        base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                        #base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], -1, 4)[base_pos_inds.type(torch.bool),bbox_label_base]

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
                    base_pos_bbox_pred, 
                    bbox_label_base,
                    classes_eq,
                    transform_applied,
                    gt_labels_aug,
                    min_size,
                    base_bbox_pred,
                    gt_base_bbox,
                    reduction_override=reduction_override)
        
            
            if 'loss_cosine' in losses.keys():
                if losses['loss_cosine'] == -1:
                    print(f'no loss')
                    losses = dict()
        return losses
   
    @force_fp32(apply_to=('cont_feat'))
    def loss_contrast_cls(self,
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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
        else:
            if self.queue_res is not None and self.use_queue:
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                
                losses['loss_contrastive_cls'], pred_cont = self.contrast_loss_classif(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    queue_res, 
                    queue_trg, 
                    queue_iou,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    reduction_override=reduction_override)
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
        
        return losses
   
@HEADS.register_module()
class QueueAugContrastiveBBoxHead_Branch_classqueue_replace_withbg(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 loss_all = None,
                 contrast_loss_classif = None,
                 to_norm_cls = False,
                 main_training = False,
                 same_class = False,
                 same_class_all = False,
                 queue_path = 'init_queue.p',
                 init_queue = False,
                 use_queue = False,
                 use_base_queue = True,
                 use_novel_queue = True,
                 use_aug_queue = True,
                 queue_length = 60,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_all is None:
            if loss_c_cls is not None:
                self.contrast_loss_cls = build_loss(loss_c_cls)
            else:
                self.contrast_loss_cls = None

            if loss_cosine is not None:
                self.contrast_loss = build_loss(loss_cosine)
            else:
                self.contrast_loss = None
            if loss_base_aug is not None:
                self.contrast_loss_base_aug = build_loss(loss_base_aug)
            else:
                self.contrast_loss_base_aug = None
            if contrast_loss_classif is not None:
                self.contrast_loss_classif = build_loss(contrast_loss_classif)
            else:
                self.contrast_loss_classif = None
            self.contrast_loss_all = None
        else:
            self.contrast_loss_all = build_loss(loss_all)
            self.contrast_loss_cls = None
            self.contrast_loss = None
            self.contrast_loss_base_aug = None


        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.queue_length = queue_length
        if self.use_queue and not init_queue:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = self.load_queue_class(use_base_queue, use_novel_queue, use_aug_queue)

            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', queue_base_ptr)
        else:
            self.queue_res = None

        self.to_norm = to_norm_cls


        self.same_class_all = same_class_all
        if self.same_class_all:
            self.same_class = False
        else: 
            self.same_class = same_class
        self.main_training = main_training
        
        self.id_save_FT = 1
        self.init_queue = init_queue
        if init_queue:
            self.queue_new_res = torch.empty((self.queue_length, mlp_head_channels))
            self.queue_new_trg = -torch.ones((self.queue_length))
            self.queue_new_iou = torch.ones((self.queue_length))


    def queue_init(self, new_dict):
        if self.queue_res is None:
            self.queue_res = new_dict
        else:
            z = self.queue_res.copy()   
            z.update(new_dict)
            self.queue_res = z

        with open(self.queue_path, 'wb') as fp:
            pickle.dump(self.queue_res, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @torch.no_grad()
    def update_queue_class(self, new_base, new_novel, new_aug):
        
        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
            fs = new_base[0].shape[1]
            cur_device = new_base[0].device
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
            fs = new_novel[0].shape[1]
            cur_device = new_novel[0].device
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
            fs = new_aug[0].shape[1]
            cur_device = new_aug[0].device
        else:
            assert True, " One of base, novel or aug should not be none"

        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1
        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        if queue_base_trg is not None:
            update = True
            ptr = self.queue_ptr.int() 
            reset = False
            increase = False

            for class_id in range(self.num_classes+1):
                tmp = None
                if new_base is not None:
                    if class_id in new_base[1].unique():
                        if ptr[class_id] == (self.queue_length//self.num_classes):
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+((self.num_classes+1)*ptr[class_id])] == -1:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                            #ptr[class_id] += 1
                        else:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                            #ptr[class_id] += 1
                            
                if new_novel is not None:
                    if class_id in new_novel[1].unique():
                        if tmp is not None:
                            tmp2 = new_novel[0][new_novel[1][new_novel[1] == class_id]]

                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes):
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+((self.num_classes+1)*ptr[class_id])] == -1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True



                if new_aug is not None:
                    if class_id in new_aug[1].unique():
                        if tmp is not None:
                            tmp2 = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes):
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+((self.num_classes+1)*ptr[class_id])] == -1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True

                if tmp is not None:
                    if reset:
                        ptr[class_id] = 0
                    tmp = tmp.mean(dim=0)
                    queue_base_res[class_id+((self.num_classes+1)*ptr[class_id])] = tmp
                    queue_base_trg[class_id+((self.num_classes+1)*ptr[class_id])] = class_id
                    
                    if reset and increase:
                        assert True, "problem, reset and increase"
                    elif increase:
                        ptr[class_id] += 1
            self.queue_res = queue_base_res.to(cur_device).detach()
            self.queue_trg = queue_base_trg.to(cur_device).detach()
            self.queue_ptr = ptr
            self.queue_iou = queue_base_iou.to(cur_device).detach()
            
        else:                
            shape_queue = (self.queue_length//(self.num_classes+1))*(self.num_classes+1)
            queue_new_res = torch.empty((shape_queue, fs))
            queue_new_trg = -torch.ones((shape_queue))
            queue_new_iou = torch.ones((shape_queue))
            len_id = 0
            if new_base is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_base[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_novel is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_novel[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_novel[0][new_novel[1][new_novel[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_aug is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_aug[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_aug[0][new_aug[1][new_aug[1] == class_id]].mean(dim=0)


            
            self.queue_res = queue_new_res.to(cur_device).detach()
            self.queue_trg = queue_new_trg.to(cur_device).detach()
            self.queue_iou = queue_new_iou.to(cur_device).detach()
    
    def load_queue_class_new(self, base=True, novel=True, aug=True):
        
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
            bg_base = -1
            
            
            if 'bg' in data.keys():
                bg_base = data['bg']['trg'].item()
            for k in data.keys():
                key_type = type(k)
                break
            
            
            for key_class in range(len(data.keys())):#self.num_classes):
                if bg_base == key_class:
                    queue_res.append(data['bg'])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                else:
                    if key_type is str:
                        if type(data[str(key_class)]) is Dict:
                            queue_res.append(data[str(key_class)]['results'])
                        else:
                            queue_res.append(data[str(key_class)])
                    else:
                        if type(data[key_class]) is Dict:
                            queue_res.append(data[key_class]['results'])
                        else:
                            queue_res.append(data[key_class])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                queue_ptr.append(1.)
            queue_new_ptr = torch.tensor(queue_ptr).long()
            if (len(data.keys())) < (self.num_classes+1):
                queue_base_ptr = torch.zeros(self.num_classes+1, dtype=torch.long)
                queue_base_ptr[:queue_new_ptr.shape[0]] = queue_new_ptr.clone()
            else:
                queue_base_ptr = queue_new_ptr.clone()

            queue_new_res = torch.stack(queue_res)
            queue_new_trg = torch.tensor(queue_trg)
            queue_new_iou = torch.tensor(queue_iou)

            fs = queue_new_res.shape[1]
            shape_queue = (self.queue_length//(self.num_classes+1))*(self.num_classes+1)

            queue_base_res = torch.zeros((shape_queue, fs))
            queue_base_res[:queue_new_res.shape[0]] = queue_new_res.clone()

            queue_base_trg = -torch.ones((shape_queue))
            queue_base_trg[:queue_new_trg.shape[0]] = queue_new_trg.clone()

            queue_base_iou = torch.ones((shape_queue))
            queue_base_iou[:queue_new_iou.shape[0]] = queue_new_iou.clone()

        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes+1, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

    def load_queue_class(self, base=True, novel=True, aug=True):
        if self.queue_path is not None:
            if exists(self.queue_path):
                with open(self.queue_path, 'rb') as fp:
                    data = pickle.load(fp)
                queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []

                for key_class in range(self.num_classes+1):
                    if str(key_class) not in data.keys():
                        queue_res.append(torch.zeros(128))
                        queue_trg.append(-1)
                        queue_iou.append(-1.)
                        queue_ptr.append(0.)
                    else:
                        if len(data[str(key_class)].shape) > 1:
                            tmp_res, tmp_trg, tmp_iou = [], [], []
                            tmp_ptr = 0
                            for rep_id in range(data[str(key_class)].shape[0]):
                                tmp_res.append(data[str(key_class)][rep_id])
                                tmp_trg.append(key_class)
                                tmp_iou.append(1.)
                                tmp_ptr += 1.
                            queue_res.append(tmp_res)
                            queue_trg.append(tmp_trg)
                            queue_iou.append(tmp_iou)
                            queue_ptr.append(tmp_ptr)
                            del tmp_res, tmp_trg, tmp_iou
                        else:
                            queue_res.append(data[str(key_class)])
                            queue_trg.append(key_class)
                            queue_iou.append(1.)
                            queue_ptr.append(1.)
                queue_new_ptr = torch.tensor(queue_ptr).long()
                queue_base_ptr = queue_new_ptr.clone()

                fs = 128
                shape_queue = (self.queue_length//(self.num_classes+1))*(1+self.num_classes)
                nbr_rep_in_queue = self.queue_length//(self.num_classes+1)
                queue_base_res = torch.zeros((shape_queue, fs))
                queue_base_trg = -torch.ones((shape_queue))
                queue_base_iou = torch.ones((shape_queue))
                for rep_id in range(queue_base_ptr.shape[0]):
                    if queue_base_ptr[rep_id] > 1.:
                        for nbr_rep_id in range(len(queue_res[rep_id])):
                            if nbr_rep_id < nbr_rep_in_queue:
                                queue_base_res[rep_id+nbr_rep_id*(self.num_classes+1)] = queue_res[rep_id][nbr_rep_id]
                                queue_base_trg[rep_id+nbr_rep_id*(self.num_classes+1)] = queue_trg[rep_id][nbr_rep_id]
                                queue_base_iou[rep_id+nbr_rep_id*(self.num_classes+1)] = queue_iou[rep_id][nbr_rep_id]
                    else:
                        if type(queue_res[rep_id]) == list:
                            queue_new_res = queue_res[rep_id][0]
                            queue_new_trg = queue_trg[rep_id][0]
                            queue_new_iou = queue_iou[rep_id][0]
                        else:
                            queue_new_res = queue_res[rep_id]
                            queue_new_trg = queue_trg[rep_id]
                            queue_new_iou = queue_iou[rep_id]

                        queue_base_res[rep_id] = queue_new_res#.clone()
                        queue_base_trg[rep_id] = queue_new_trg#.clone()
                        queue_base_iou[rep_id] = queue_new_iou#.clone()
            else:
                queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes+1, dtype=torch.long)
        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes+1, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

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
            cls_score, bbox_pred, contrast_feat  = self.forward_one_branch(x, True)
            aug_cls_score, aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug, True)
        else:
            cls_score, bbox_pred, aug_cls_score, aug_bbox_pred = None, None, None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_cls_score, base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_cls_score, base_bbox_pred, base_contrast_feat, cls_score, bbox_pred, contrast_feat, aug_cls_score, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x, base=True):
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

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # contrastive branch

        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)

        
        if self.to_norm:
        # cls branch
            if x_cls.dim() > 2:
                x_cls = torch.flatten(x_cls, start_dim=1)

            # normalize the input x along the `input_size` dimension
            x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_cls_normalized = x_cls.div(x_norm + self.eps).clone()
            # normalize weight
            with torch.no_grad():
                temp_norm = torch.norm(self.fc_cls.weight, p=2,dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
                self.fc_cls.weight.clone().div_(temp_norm + self.eps)
            # calculate and scale cls_score
            
            if base: 
                cls_output = self.fc_cls(x_cls_normalized)
                cls_score = self.scale * cls_output if self.with_cls else None
            else:
                with torch.no_grad():
                    cls_output = self.fc_cls(x_cls_normalized)
                    cls_score = self.scale * cls_output if self.with_cls else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None

        
        return cls_score, bbox_pred, cont_feat


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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}

            if new_base is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
            losses = dict()
        else:
            if self.main_training and not self.same_class and not self.use_queue:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            elif self.main_training and self.same_class and self.use_queue:
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1   
            elif self.main_training and self.same_class_all and self.use_queue:
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                
                self.id_save_FT += 1   
            
            elif self.main_training and self.same_class:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override) 
            elif self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None
                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                # base novel aug
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                
                
                self.id_save_FT += 1          
            else:

                if base_proposal_ious is not None:

                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                else:

                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            bbox_results['cont_feat'], 
                            gt_base_bbox[0], 
                            bbox_targets[0],
                            classes_eq, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            aug_bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            aug_bbox_targets[0], 
                            None, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
        
            if self.contrast_loss_bbox is not None:
                pos_inds_aug = (aug_bbox_targets[0] >= 0) & (aug_bbox_targets[0] > self.num_classes) 
                pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] > self.num_classes)
                base_pos_inds = (gt_base_bbox[0] >= 0) & (gt_base_bbox[0] < self.num_classes)
                
                
                # do not perform bounding box regression for BG anymore.
                
                if pos_inds.any() and pos_inds_aug.any():
                    # novel
                    if self.reg_class_agnostic:
                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
                    else:
                        bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                        labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], -1, 4)
                        pos_bbox_pred = pos_bbox_pred[pos_inds.type(torch.bool),labels_true]

                    # aug
                    if self.reg_class_agnostic:
                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
                    else:
                        bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                        labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], -1,
                            4)[pos_inds_aug.type(torch.bool),labels_true_aug]
                    # base
                    if self.reg_class_agnostic:
                        base_pos_bbox_pred = gt_base_bbox[2].view(
                            gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                    else:
                        bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                        base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                        #base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], -1, 4)[base_pos_inds.type(torch.bool),bbox_label_base]

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
                    base_pos_bbox_pred, 
                    bbox_label_base,
                    classes_eq,
                    transform_applied,
                    gt_labels_aug,
                    min_size,
                    base_bbox_pred,
                    gt_base_bbox,
                    reduction_override=reduction_override)
        
            
            if 'loss_cosine' in losses.keys():
                if losses['loss_cosine'] == -1:
                    print(f'no loss')
                    losses = dict()
        return losses
   
    @force_fp32(apply_to=('cont_feat'))
    def loss_contrast_cls(self,
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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
        else:
            if self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                
                losses['loss_contrastive_cls'], pred_cont = self.contrast_loss_classif(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    queue_res, 
                    queue_trg, 
                    queue_iou,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    reduction_override=reduction_override)
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
        
        return losses
   

@HEADS.register_module()
class Agnostic_BBoxHead_Branch(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 to_norm_cls = False,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        

    def forward(self, x):
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
        bbox_pred  = self.forward_one_branch(x)
        
        
        return bbox_pred


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
        x_reg = x

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

            
        return bbox_pred


    def set_decay_rate(self, decay_rate: float) -> None:
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.

        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

@HEADS.register_module()
class Agnostic_QueueAugContrastiveBBoxHead_Branch(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 loss_all = None,
                 contrast_loss_classif = None,
                 to_norm_cls = False,
                 main_training = False,
                 same_class = False,
                 same_class_all = False,
                 queue_path = 'init_queue.p',
                 use_queue = False,
                 use_base_queue = True,
                 use_novel_queue = True,
                 use_aug_queue = True,
                 queue_length = 60,
                 num_proposals = 256,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_all is None:
            if loss_c_cls is not None:
                self.contrast_loss_cls = build_loss(loss_c_cls)
            else:
                self.contrast_loss_cls = None

            if loss_cosine is not None:
                self.contrast_loss = build_loss(loss_cosine)
            else:
                self.contrast_loss = None
            if loss_base_aug is not None:
                self.contrast_loss_base_aug = build_loss(loss_base_aug)
            else:
                self.contrast_loss_base_aug = None
            if contrast_loss_classif is not None:
                self.contrast_loss_classif = build_loss(contrast_loss_classif)
            else:
                self.contrast_loss_classif = None
            self.contrast_loss_all = None
        else:
            self.contrast_loss_all = build_loss(loss_all)
            self.contrast_loss_cls = None
            self.contrast_loss = None
            self.contrast_loss_base_aug = None


        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.save_bbox_feat = False
        self.num_proposals = num_proposals
        self.queue_length = queue_length
        if self.use_queue:
            queue_base_res, queue_base_trg, queue_base_iou = self.load_queue_3(use_base_queue, use_novel_queue, use_aug_queue)
            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_trgX', queue_base_trg)

            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('queue_ptr_X', torch.zeros(1, dtype=torch.long))

        else:
            self.queue_res = None

        self.to_norm = to_norm_cls

        self.same_class = same_class
        self.same_class_all = same_class_all
        self.main_training = main_training

        self.id_save_FT = 1
            
    def queue_init(self, new_dict):
        if self.queue_res is None:
            self.queue_res = new_dict
        else:
            z = self.queue_res.copy()   
            z.update(new_dict)
            self.queue_res = z

        with open(self.queue_path, 'wb') as fp:
            pickle.dump(self.queue_res, fp, protocol=pickle.HIGHEST_PROTOCOL)


    @torch.no_grad()
    def update_queue(self, new_base, new_novel, new_aug):

        # new X: feat, lbl, iou 
        #   Feat : Ql x N x D
        #   lbl : Ql x N
        #   iou : Ql x N x D


        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        queue_length_frac = int(queue_base_res.shape[0]/3)
        
        bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1

        if queue_base_res.shape[0] < self.queue_length:

            queue_new_res = torch.cat((queue_base_res[:queue_length_frac], torch.unsqueeze(new_base[0], dim=0), 
                                        queue_base_res[queue_length_frac:2*queue_length_frac], torch.unsqueeze(new_novel[0], dim=0), 
                                        queue_base_res[2*queue_length_frac:], torch.unsqueeze(new_aug[0], dim=0))) 
            queue_new_trg = torch.cat([queue_base_trg[:queue_length_frac], torch.unsqueeze(new_base[1], dim=0), 
                                        queue_base_trg[queue_length_frac:2*queue_length_frac], torch.unsqueeze(new_novel[1], dim=0), 
                                        queue_base_trg[2*queue_length_frac:], torch.unsqueeze(new_aug[1], dim=0)]) 
            queue_new_iou = torch.cat([queue_base_iou[:queue_length_frac], torch.unsqueeze(new_base[2], dim=0), 
                                        queue_base_iou[queue_length_frac:2*queue_length_frac], torch.unsqueeze(new_novel[2], dim=0), 
                                        queue_base_iou[2*queue_length_frac:], torch.unsqueeze(new_aug[2], dim=0)]) 

            self.queue_res = queue_new_res.detach()
            self.queue_trg = queue_new_trg.detach()
            self.queue_iou = queue_new_iou.detach()

        else:
            ptr = int(self.queue_ptr) 
            self.queue_res[ptr,:] = new_base[0].detach()
            self.queue_res[queue_length_frac+ptr,:] = new_novel[0].detach()
            self.queue_res[2*queue_length_frac+ptr,:] = new_aug[0].detach()

            self.queue_trg[ptr,:] = new_base[1].detach()
            self.queue_trg[queue_length_frac+ptr,:] = new_novel[1].detach()
            self.queue_trg[2*queue_length_frac+ptr,:] = new_aug[1].detach()

            self.queue_iou[ptr,:] = new_base[2].detach()
            self.queue_iou[queue_length_frac+ptr,:] = new_novel[2].detach()
            self.queue_iou[2*queue_length_frac+ptr,:] = new_aug[2].detach()

            ptr = (ptr + 1) % queue_length_frac  # move pointer

            self.queue_ptr[0] = ptr


    @torch.no_grad()
    def update_queue_2(self, new_base, new_novel, new_aug):

        
        
        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
        else:
            assert True, " One of base, novel or aug should not be none"
        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1

        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        
        ptr = int(self.queue_ptr_X) 

        if queue_base_trg is not None:
            queue_length_frac = int(queue_base_res.shape[0]/len_id)

            len_inc = 0
            if new_base is not None:
                len_inc += 1
            if new_novel is not None:
                len_inc += 1
            if new_aug is not None:
                len_inc += 1
            if queue_base_res.shape[0] <= (self.queue_length-len_inc):
                
                queue_new_res = []
                queue_new_trg = []
                queue_new_iou = []
                len_id = 0
                if new_base is not None:
                    queue_new_res += [queue_base_res[:queue_length_frac], torch.unsqueeze(new_base[0], dim=0)]
                    queue_new_trg += [queue_base_trg[:queue_length_frac], torch.unsqueeze(new_base[1], dim=0)]
                    queue_new_iou += [queue_base_iou[:queue_length_frac], torch.unsqueeze(new_base[2], dim=0)]
                    len_id += 1
                if new_novel is not None:
                    queue_new_res += [queue_base_res[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[0], dim=0).to(new_base[0].device)]
                    queue_new_trg += [queue_base_trg[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[1], dim=0).to(new_base[1].device)]
                    queue_new_iou += [queue_base_iou[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[2], dim=0).to(new_base[2].device)]
                    len_id += 1
                if new_aug is not None:
                    queue_new_res += [queue_base_res[len_id * queue_length_frac:], torch.unsqueeze(new_aug[0], dim=0).to(new_base[0].device)]
                    queue_new_trg += [queue_base_trg[len_id * queue_length_frac:], torch.unsqueeze(new_aug[1], dim=0).to(new_base[1].device)]
                    queue_new_iou += [queue_base_iou[len_id * queue_length_frac:], torch.unsqueeze(new_aug[2], dim=0).to(new_base[2].device)]

                queue_new_res = torch.cat(queue_new_res) 
                queue_new_trg = torch.cat(queue_new_trg) 
                queue_new_iou = torch.cat(queue_new_iou) 

                

                self.queue_res = queue_new_res.detach()
                self.queue_trg = queue_new_trg.detach()
                self.queue_iou = queue_new_iou.detach()

            else:
                update = True
                len_id = 0
                ptr = int(self.queue_ptr_X) 
                
                if new_base is not None:
                    if new_base[0].shape[0] != self.queue_res[ptr,:].shape[0]:
                        tmp = torch.zeros_like(self.queue_res[ptr])
                        tmp[:new_base[0].shape[0]] = new_base[0].detach()
                        self.queue_res[ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_trg[ptr])
                        tmp[:new_base[1].shape[0]] = new_base[1].detach()
                        self.queue_trg[ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_iou[ptr])
                        tmp[:new_base[2].shape[0]] = new_base[2].detach()
                        self.queue_iou[ptr,:] = tmp
                        del tmp
                    else:
                        self.queue_res[ptr,:] = new_base[0].detach()
                        self.queue_trg[ptr,:] = new_base[1].detach()
                        self.queue_iou[ptr,:] = new_base[2].detach()
                    len_id += 1
                queue_shape = self.queue_trg[len_id * queue_length_frac+ptr,:].shape[0]

                if new_novel is not None:
                    if (new_novel[0].shape[0] != queue_shape) or (new_novel[1].shape[0] != queue_shape) or (new_novel[2].shape[0] != queue_shape):
                        tmp = torch.zeros_like(self.queue_res[ptr])
                        tmp[:new_novel[0].shape[0]] = new_novel[0].detach()
                        
                        self.queue_res[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_trg[ptr])
                        tmp[:new_novel[1].shape[0]] = new_novel[1].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_iou[ptr])
                        tmp[:new_novel[2].shape[0]] = new_novel[2].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = tmp
                        len_id += 1
                        del tmp
                        #update = False
                    else:
                        self.queue_res[len_id * queue_length_frac+ptr,:] = new_novel[0].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = new_novel[1].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = new_novel[2].detach()
                        len_id += 1
                if new_aug is not None:
                    if (new_aug[0].shape[0] != queue_shape) or (new_aug[1].shape[0] != queue_shape) or (new_aug[2].shape[0] != queue_shape):
                        tmp = torch.zeros_like(self.queue_res[ptr])
                        tmp[:new_aug[0].shape[0]] = new_aug[0].detach()
                        
                        self.queue_res[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_trg[ptr])
                        tmp[:new_aug[1].shape[0]] = new_aug[1].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = tmp
                        tmp = torch.zeros_like(self.queue_iou[ptr])
                        tmp[:new_aug[2].shape[0]] = new_aug[2].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = tmp
                        len_id += 1
                        del tmp
                        #update = False
                    else:
                        self.queue_res[len_id * queue_length_frac+ptr,:] = new_aug[0].detach()
                        self.queue_trg[len_id * queue_length_frac+ptr,:] = new_aug[1].detach()
                        self.queue_iou[len_id * queue_length_frac+ptr,:] = new_aug[2].detach()
                        len_id += 1
                
                if update:

                    ptr = (ptr + len_id) % queue_length_frac

                    self.queue_ptr[0] = ptr
                    self.queue_ptr_X[0] = ptr
        else:                
            queue_new_res = []
            queue_new_trg = []
            queue_new_iou = []
            len_id = 0
            if new_base is not None:
                queue_new_res += [torch.unsqueeze(new_base[0], dim=0)]
                queue_new_trg += [torch.unsqueeze(new_base[1], dim=0)]
                queue_new_iou += [torch.unsqueeze(new_base[2], dim=0)]
                len_id += 1
            if new_novel is not None:
                queue_new_res += [torch.unsqueeze(new_novel[0], dim=0).to(new_base[0].device)]
                queue_new_trg += [torch.unsqueeze(new_novel[1], dim=0).to(new_base[1].device)]
                queue_new_iou += [torch.unsqueeze(new_novel[2], dim=0).to(new_base[2].device)]
                len_id += 1
            if new_aug is not None:
                queue_new_res += [torch.unsqueeze(new_aug[0], dim=0).to(new_base[0].device)]
                queue_new_trg += [torch.unsqueeze(new_aug[1], dim=0).to(new_base[1].device)]
                queue_new_iou += [torch.unsqueeze(new_aug[2], dim=0).to(new_base[2].device)]

            queue_new_res = torch.cat(queue_new_res) 
            queue_new_trg = torch.cat(queue_new_trg) 
            queue_new_iou = torch.cat(queue_new_iou) 
            
            self.queue_res = queue_new_res.detach()
            self.queue_trg = queue_new_trg.detach()
            self.queue_iou = queue_new_iou.detach()
        if self.save_bbox_feat:
            self.save_queue(new_base, new_novel, new_aug)
    
    def save_queue(self,new_base, new_novel, new_aug):
        reshaped_trg = self.queue_trg.clone().reshape(-1)
        reshaped_iou = self.queue_iou.clone().reshape(-1)
        reshaped_res = self.queue_res.clone().reshape(-1, self.queue_res.shape[2])
        queue_dict = {}
        for class_id in range(self.num_classes+1):
            if class_id in reshaped_trg.unique():
                msk = reshaped_trg == class_id
                res_msk = reshaped_res[msk]
                iou_msk = reshaped_iou[msk]
                trg_msk = reshaped_trg[msk]
                tmp_dict ={'results':res_msk, 'trg':trg_msk, 'iou':iou_msk} 
                queue_dict[class_id]=tmp_dict
        with open(self.queue_path, 'wb') as fp:
            pickle.dump(queue_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            self.save_bbox_feat = False

        

    def load_queue(self, base=True, novel=True, aug=True):

        with open(self.queue_path, 'rb') as fp:
            data = pickle.load(fp)
        queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg = [], [], [], [], [], []
        queue_base_iou, queue_novel_iou, queue_aug_iou = [], [], []
        for key in data.keys():
            queue_base_res.append(data[key]['base_results'])
            queue_base_trg.append(data[key]['base_trg'])
            queue_novel_res.append(data[key]['novel_results'])
            queue_novel_trg.append(data[key]['novel_trg'])
            queue_aug_res.append(data[key]['aug_results'])
            queue_aug_trg.append(data[key]['aug_trg'])
            queue_base_iou.append(data[key]['base_ious'])
            queue_novel_iou.append(data[key]['novel_ious'])
            queue_aug_iou.append(data[key]['aug_ious'])
        queue_base_res = queue_base_res + queue_novel_res + queue_aug_res
        queue_base_res = torch.stack(queue_base_res)

        queue_base_trg = queue_base_trg + queue_novel_trg + queue_aug_trg
        queue_base_trg = torch.stack(queue_base_trg)

        queue_base_iou = queue_base_iou + queue_novel_iou + queue_aug_iou
        queue_base_iou = torch.stack(queue_base_iou)

        id_shuffle = np.arange(queue_base_res.shape[0])
        np.random.shuffle(id_shuffle)
        queue_base_res = queue_base_res[id_shuffle]
        queue_base_trg = queue_base_trg[id_shuffle]
        queue_base_iou = queue_base_iou[id_shuffle]
        
        return queue_base_res, queue_base_trg, queue_base_iou

    def load_queue_2(self, base=True, novel=True, aug=True):
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou = [], [], []
            for key in data.keys():
                if base:
                    queue_res.append(data[key]['base_results'])
                    queue_trg.append(data[key]['base_trg'])
                    queue_iou.append(data[key]['base_ious'])

                if novel:  
                    queue_res.append(data[key]['novel_results'])
                    queue_trg.append(data[key]['novel_trg'])
                    queue_iou.append(data[key]['novel_ious'])

                if aug:
                    queue_res.append(data[key]['aug_results'])
                    queue_trg.append(data[key]['aug_trg'])
                    queue_iou.append(data[key]['aug_ious'])


            queue_base_res = torch.stack(queue_res)
            queue_base_trg = torch.stack(queue_trg)
            queue_base_iou = torch.stack(queue_iou)


            id_shuffle = np.arange(queue_base_res.shape[0])
            np.random.shuffle(id_shuffle)
            queue_base_res = queue_base_res[id_shuffle]
            queue_base_trg = queue_base_trg[id_shuffle]
            queue_base_iou = queue_base_iou[id_shuffle]
        else:
            queue_base_res, queue_base_trg, queue_base_iou = torch.zeros(1), None, torch.zeros(1)

        return queue_base_res, queue_base_trg, queue_base_iou
    
    def load_queue_3(self, base=True, novel=True, aug=True):
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou = [], [], []
            for key in data.keys():
                for second_key in data[key].keys():
                    if 'results' in second_key:
                        queue_res.append(data[key][second_key])
                    if 'trg' in second_key:
                        queue_trg.append(data[key][second_key])
                    if 'iou' in second_key:
                        queue_iou.append(data[key][second_key])
            queue_base_res = torch.vstack(queue_res).reshape(-1, self.num_proposals, 128)
            queue_base_trg = torch.hstack(queue_trg).reshape(-1, self.num_proposals)
            queue_base_iou = torch.hstack(queue_iou).reshape(-1, self.num_proposals)
            
            # .reshape(-1, self.num_proposals, 128)
            id_shuffle = np.arange(queue_base_res.shape[0])
            np.random.shuffle(id_shuffle)
            queue_base_res = queue_base_res[id_shuffle]
            queue_base_trg = queue_base_trg[id_shuffle]
            queue_base_iou = queue_base_iou[id_shuffle]
        else:
            queue_base_res, queue_base_trg, queue_base_iou = torch.zeros(1), None, torch.zeros(1)

        return queue_base_res, queue_base_trg, queue_base_iou

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
            bbox_pred, contrast_feat  = self.forward_one_branch(x, True)
            aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug, True)
        else:
            bbox_pred, aug_bbox_pred = None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_bbox_pred, base_contrast_feat, bbox_pred, contrast_feat, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x, base=True):
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
        x_reg = x
        x_contra = x

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None


        # contrastive branch
        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)

        return bbox_pred, cont_feat




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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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
        if self.main_training and not self.same_class and not self.use_queue:
            if self.with_weight_decay:
                decay_rate = self._decay_rate
            if self.contrast_loss_base_aug is not None:

                losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)   
            if self.contrast_loss_cls is not None:
                losses['loss_c_cls'] = self.contrast_loss_cls(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss is not None:
                losses['loss_cosine'] = self.contrast_loss(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss_all is not None:
                losses['loss_contrastive'] = self.contrast_loss_all(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    queue_res, 
                    queue_trg, 
                    queue_iou,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
        elif self.main_training and self.same_class_all and self.use_queue:
            if self.queue_trg is None:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            else:
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            classes_eq[num_classes]=num_classes
            if self.use_base_queue:
                if (gt_base_bbox[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                else:
                    queue_trg = gt_base_bbox[0]
                base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
            else:
                base_update = None
            if self.use_novel_queue:
                if (bbox_targets[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                else:
                    queue_trg = bbox_targets[0]
                novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
            else:
                novel_update = None

            if self.use_aug_queue:
                if (aug_bbox_targets[0] > num_classes).any():
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                else:
                    queue_trg = aug_bbox_targets[0]
                aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
            else:
                aug_update = None

            self.update_queue_2(base_update, novel_update, aug_update )
            self.id_save_FT += 1   
        elif self.main_training and self.same_class and self.use_queue:
            if self.queue_trg is None:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            else:
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            classes_eq[num_classes]=num_classes
            if self.use_base_queue:
                if (gt_base_bbox[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                else:
                    queue_trg = gt_base_bbox[0]
                base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
            else:
                base_update = None
            if self.use_novel_queue:
                if (bbox_targets[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                else:
                    queue_trg = bbox_targets[0]
                novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
            else:
                novel_update = None

            if self.use_aug_queue:
                if (aug_bbox_targets[0] > num_classes).any():
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                else:
                    queue_trg = aug_bbox_targets[0]
                aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
            else:
                aug_update = None

            self.update_queue_2(base_update, novel_update, aug_update )
            self.id_save_FT += 1   
        elif self.main_training and self.same_class and not self.use_queue:
            if self.with_weight_decay:
                decay_rate = self._decay_rate
            if self.contrast_loss_base_aug is not None:

                losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)   
            if self.contrast_loss_cls is not None:
                losses['loss_c_cls'] = self.contrast_loss_cls(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss is not None:
                losses['loss_cosine'] = self.contrast_loss(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
            if self.contrast_loss_all is not None:
                losses['loss_contrastive'] = self.contrast_loss_all(
                    base_bbox_pred['cont_feat'], 
                    gt_base_bbox[0], 
                    base_proposal_ious,
                    bbox_results['cont_feat'], 
                    bbox_targets[0],
                    proposal_ious,
                    aug_bbox_results['cont_feat'], 
                    aug_bbox_targets[0], 
                    aug_proposal_ious,
                    classes_equi = None, 
                    nbr_classes = num_classes,
                    decay_rate=decay_rate,
                    reduction_override=reduction_override)
        elif self.queue_res is not None and self.use_queue:
            #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
            
            # base novel aug
            if self.queue_trg is None:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            else:
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], # cont_feat
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                
            
            classes_eq[num_classes]=num_classes
            if self.use_base_queue:
                if (gt_base_bbox[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                else:
                    queue_trg = gt_base_bbox[0]
                base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
            else:
                base_update = None
            if self.use_novel_queue:
                if (bbox_targets[0] > num_classes).any():                    
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                else:
                    queue_trg = bbox_targets[0]
                novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
            else:
                novel_update = None

            if self.use_aug_queue:
                if (aug_bbox_targets[0] > num_classes).any():
                    queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                else:
                    queue_trg = aug_bbox_targets[0]
                aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
            else:
                aug_update = None

            self.update_queue_2(base_update, novel_update, aug_update )
            self.id_save_FT += 1   
        
        else:

            if base_proposal_ious is not None:

                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:
                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            else:

                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        bbox_results['cont_feat'], 
                        gt_base_bbox[0], 
                        bbox_targets[0],
                        classes_eq, 
                        num_classes,
                        reduction_override=reduction_override)
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        aug_bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        aug_bbox_targets[0], 
                        None, 
                        num_classes,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
    
        if self.contrast_loss_bbox is not None and False:
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
        
        if self.contrast_loss_bbox is not None:
            pos_inds_aug = (aug_bbox_targets[0] >= 0) & (aug_bbox_targets[0] > self.num_classes) 
            pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] > self.num_classes)
            base_pos_inds = (gt_base_bbox[0] >= 0) & (gt_base_bbox[0] < self.num_classes)
            
            # do not perform bounding box regression for BG anymore.
            
            if pos_inds.any() and pos_inds_aug.any():
                # novel
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_score_pred.view(
                        bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
                else:
                    bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                    labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                    pos_bbox_pred = bbox_score_pred.view(
                        bbox_score_pred.shape[0], -1, 4)[pos_inds.type(torch.bool),labels_true]
                # aug
                if self.reg_class_agnostic:
                    aug_pos_bbox_pred = aug_bbox_score_pred.view(
                        aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
                else:
                    bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                    labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                    aug_pos_bbox_pred = aug_bbox_score_pred.view(
                        aug_bbox_score_pred.shape[0], -1,
                        4)[pos_inds_aug.type(torch.bool),labels_true_aug]
                # base
                if self.reg_class_agnostic:
                    base_pos_bbox_pred = base_bbox_pred['bbox_pred'].view(
                        base_bbox_pred['bbox_pred'].shape[0], 4)[base_pos_inds.type(torch.bool)]
                else:
                    bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                    base_pos_bbox_pred = base_bbox_pred['bbox_pred'].view(
                        base_bbox_pred['bbox_pred'].shape[0], -1,
                        4)[base_pos_inds.type(torch.bool),bbox_label_base]
                
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
                base_pos_bbox_pred, 
                bbox_label_base,
                classes_eq,
                transform_applied,
                gt_labels_aug,
                min_size,
                base_bbox_pred,
                gt_base_bbox,
                reduction_override=reduction_override)
        
        if not self.use_queue and not self.main_training:
            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)
            if (aug_bbox_targets[0]> num_classes).any():
                aug_bbox_save = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0]]).to(aug_bbox_results['cont_feat'].device)

            cont_dict = {
                'base_results': base_bbox_pred['cont_feat'], 
                'base_trg': gt_base_bbox[0], 
                'base_ious': base_proposal_ious,
                'novel_results': bbox_results['cont_feat'], 
                'novel_trg': bbox_save, 
                'novel_ious': proposal_ious,
                'aug_results': aug_bbox_results['cont_feat'], 
                'aug_trg': aug_bbox_save,
                'aug_ious': aug_proposal_ious
                }
            id_dict = {str(self.id_save_FT):cont_dict}
            self.id_save_FT += 1
            
            self.queue_init(id_dict)
        if 'loss_cosine' in losses.keys():
            if losses['loss_cosine'] == -1:
                print(f'no loss')
                losses = dict()
        return losses

@HEADS.register_module()
class Agnostic_QueueAugContrastiveBBoxHead_Branch_classqueue_replace(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 loss_all = None,
                 contrast_loss_classif = None,
                 to_norm_cls = False,
                 main_training = False,
                 same_class = False,
                 same_class_all = False,
                 queue_path = 'init_queue.p',
                 init_queue = False,
                 use_queue = False,
                 use_base_queue = True,
                 use_novel_queue = True,
                 use_aug_queue = True,
                 queue_length = 60,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_all is None:
            if loss_c_cls is not None:
                self.contrast_loss_cls = build_loss(loss_c_cls)
            else:
                self.contrast_loss_cls = None

            if loss_cosine is not None:
                self.contrast_loss = build_loss(loss_cosine)
            else:
                self.contrast_loss = None
            if loss_base_aug is not None:
                self.contrast_loss_base_aug = build_loss(loss_base_aug)
            else:
                self.contrast_loss_base_aug = None
            if contrast_loss_classif is not None:
                self.contrast_loss_classif = build_loss(contrast_loss_classif)
            else:
                self.contrast_loss_classif = None
            self.contrast_loss_all = None
        else:
            self.contrast_loss_all = build_loss(loss_all)
            self.contrast_loss_cls = None
            self.contrast_loss = None
            self.contrast_loss_base_aug = None


        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.queue_length = queue_length
        if self.use_queue and not init_queue:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = self.load_queue_class(use_base_queue, use_novel_queue, use_aug_queue)

            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', queue_base_ptr)
        else:
            self.queue_res = None

        self.to_norm = to_norm_cls


        self.same_class_all = same_class_all
        if self.same_class_all:
            self.same_class = False
        else: 
            self.same_class = same_class
        self.main_training = main_training
        
        self.id_save_FT = 1
        self.init_queue = init_queue
        if init_queue:
            self.queue_new_res = torch.empty((self.queue_length, mlp_head_channels))
            self.queue_new_trg = -torch.ones((self.queue_length))
            self.queue_new_iou = torch.ones((self.queue_length))


    def queue_init(self, new_dict):
        if self.queue_res is None:
            self.queue_res = new_dict
        else:
            z = self.queue_res.copy()   
            z.update(new_dict)
            self.queue_res = z

        with open(self.queue_path, 'wb') as fp:
            pickle.dump(self.queue_res, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @torch.no_grad()
    def update_queue_class(self, new_base, new_novel, new_aug):
        
        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
            fs = new_base[0].shape[1]
            cur_device = new_base[0].device
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
            fs = new_novel[0].shape[1]
            cur_device = new_novel[0].device
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
            fs = new_aug[0].shape[1]
            cur_device = new_aug[0].device
        else:
            assert True, " One of base, novel or aug should not be none"

        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1
        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        if queue_base_trg is not None:
            update = True
            ptr = self.queue_ptr.int() 
            reset = False
            increase = False

            for class_id in range(self.num_classes):
                tmp = None
                if new_base is not None:
                    if class_id in new_base[1].unique():
                        if ptr[class_id] == (self.queue_length//self.num_classes-1):
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                            
                if new_novel is not None:
                    if class_id in new_novel[1].unique():
                        if tmp is not None:
                            tmp2 = new_novel[0][new_novel[1][new_novel[1] == class_id]]

                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes)-1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True



                if new_aug is not None:
                    if class_id in new_aug[1].unique():
                        if tmp is not None:
                            tmp2 = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes)-1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True

                if tmp is not None:
                    tmp = tmp.mean(dim=0)
                    queue_base_res[class_id+(self.num_classes*ptr[class_id])] = tmp
                    queue_base_trg[class_id+(self.num_classes*ptr[class_id])] = class_id

                    if reset and increase:
                        assert True, "problem, reset and increase"
                    elif reset:
                        ptr[class_id] = 0
                    elif increase:
                        ptr[class_id] += 1
            self.queue_res = queue_base_res.to(cur_device).detach()
            self.queue_trg = queue_base_trg.to(cur_device).detach()
            self.queue_ptr = ptr
            self.queue_iou = queue_base_iou.to(cur_device).detach()
            
        else:                
            shape_queue = (self.queue_length//self.num_classes)*self.num_classes
            queue_new_res = torch.empty((shape_queue, fs))
            queue_new_trg = -torch.ones((shape_queue))
            queue_new_iou = torch.ones((shape_queue))
            len_id = 0
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_novel is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_novel[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_novel[0][new_novel[1][new_novel[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_aug is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_aug[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_aug[0][new_aug[1][new_aug[1] == class_id]].mean(dim=0)


            
            self.queue_res = queue_new_res.to(cur_device).detach()
            self.queue_trg = queue_new_trg.to(cur_device).detach()
            self.queue_iou = queue_new_iou.to(cur_device).detach()
    
    @torch.no_grad()
    def update_queue_class_withbg(self, new_base, new_novel, new_aug):
        
        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
            fs = new_base[0].shape[1]
            cur_device = new_base[0].device
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
            fs = new_novel[0].shape[1]
            cur_device = new_novel[0].device
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
            fs = new_aug[0].shape[1]
            cur_device = new_aug[0].device
        else:
            assert True, " One of base, novel or aug should not be none"

        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1
        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        if queue_base_trg is not None:
            update = True
            ptr = self.queue_ptr.int() 
            reset = False
            increase = False

            for class_id in range(self.num_classes+1):
                tmp = None
                if new_base is not None:
                    if class_id in new_base[1].unique():
                        if ptr[class_id] == (self.queue_length//self.num_classes-1):
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                            
                if new_novel is not None:
                    if class_id in new_novel[1].unique():
                        if tmp is not None:
                            tmp2 = new_novel[0][new_novel[1][new_novel[1] == class_id]]

                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes)-1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True



                if new_aug is not None:
                    if class_id in new_aug[1].unique():
                        if tmp is not None:
                            tmp2 = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//self.num_classes)-1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -1:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True

                if tmp is not None:
                    tmp = tmp.mean(dim=0)
                    queue_base_res[class_id+(self.num_classes*ptr[class_id])] = tmp
                    queue_base_trg[class_id+(self.num_classes*ptr[class_id])] = class_id

                    if reset and increase:
                        assert True, "problem, reset and increase"
                    elif reset:
                        ptr[class_id] = 0
                    elif increase:
                        ptr[class_id] += 1
            self.queue_res = queue_base_res.to(cur_device).detach()
            self.queue_trg = queue_base_trg.to(cur_device).detach()
            self.queue_ptr = ptr
            self.queue_iou = queue_base_iou.to(cur_device).detach()
            
        else:                
            shape_queue = (self.queue_length//self.num_classes)*self.num_classes
            queue_new_res = torch.empty((shape_queue, fs))
            queue_new_trg = -torch.ones((shape_queue))
            queue_new_iou = torch.ones((shape_queue))
            len_id = 0
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_novel is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_novel[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_novel[0][new_novel[1][new_novel[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_aug is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_aug[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_aug[0][new_aug[1][new_aug[1] == class_id]].mean(dim=0)


            
            self.queue_res = queue_new_res.to(cur_device).detach()
            self.queue_trg = queue_new_trg.to(cur_device).detach()
            self.queue_iou = queue_new_iou.to(cur_device).detach()
    
    def load_queue_class_new(self, base=True, novel=True, aug=True):
        
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
            bg_base = -1
            
            
            if 'bg' in data.keys():
                bg_base = data['bg']['trg'].item()
            for k in data.keys():
                key_type = type(k)
                break
            
            
            for key_class in range(len(data.keys())-1):#self.num_classes):
                if bg_base == key_class:
                    queue_res.append(data['bg'])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                else:
                    if key_type is str:
                        if type(data[str(key_class)]) is Dict:
                            queue_res.append(data[str(key_class)]['results'])
                        else:
                            queue_res.append(data[str(key_class)])
                    else:
                        if type(data[key_class]) is Dict:
                            queue_res.append(data[key_class]['results'])
                        else:
                            queue_res.append(data[key_class])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                queue_ptr.append(1.)
            queue_new_ptr = torch.tensor(queue_ptr).long()
            if (len(data.keys())-1) < self.num_classes:
                queue_base_ptr = torch.zeros(self.num_classes, dtype=torch.long)
                queue_base_ptr[:queue_new_ptr.shape[0]] = queue_new_ptr.clone()
            else:
                queue_base_ptr = queue_new_ptr.clone()

            queue_new_res = torch.stack(queue_res)
            queue_new_trg = torch.tensor(queue_trg)
            queue_new_iou = torch.tensor(queue_iou)

            fs = queue_new_res.shape[1]
            shape_queue = (self.queue_length//self.num_classes)*self.num_classes

            queue_base_res = torch.zeros((shape_queue, fs))
            queue_base_res[:queue_new_res.shape[0]] = queue_new_res.clone()

            queue_base_trg = -torch.ones((shape_queue))
            queue_base_trg[:queue_new_trg.shape[0]] = queue_new_trg.clone()

            queue_base_iou = torch.ones((shape_queue))
            queue_base_iou[:queue_new_iou.shape[0]] = queue_new_iou.clone()

        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

    def load_queue_class(self, base=True, novel=True, aug=True):
        
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
            
            for key_class in range(self.num_classes):
                queue_res.append(data[str(key_class)])
                queue_trg.append(key_class)
                queue_iou.append(1.)
                queue_ptr.append(1.)
            queue_new_ptr = torch.tensor(queue_ptr).long()
            queue_base_ptr = queue_new_ptr.clone()

            queue_new_res = torch.stack(queue_res)
            queue_new_trg = torch.tensor(queue_trg)
            queue_new_iou = torch.tensor(queue_iou)

            fs = queue_new_res.shape[1]
            shape_queue = (self.queue_length//self.num_classes)*self.num_classes

            queue_base_res = torch.zeros((shape_queue, fs))
            queue_base_res[:queue_new_res.shape[0]] = queue_new_res.clone()

            queue_base_trg = -torch.ones((shape_queue))
            queue_base_trg[:queue_new_trg.shape[0]] = queue_new_trg.clone()

            queue_base_iou = torch.ones((shape_queue))
            queue_base_iou[:queue_new_iou.shape[0]] = queue_new_iou.clone()

        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

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
            bbox_pred, contrast_feat  = self.forward_one_branch(x, True)
            aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug, True)
        else:
            bbox_pred, aug_bbox_pred = None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_bbox_pred, base_contrast_feat, bbox_pred, contrast_feat, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x, base=True):
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
        x_reg = x
        x_contra = x
       
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # contrastive branch

        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)
        
        return bbox_pred, cont_feat


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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}

            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
            losses = dict()
        else:
            if self.main_training and not self.same_class and not self.use_queue:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            elif self.main_training and self.same_class and self.use_queue:
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1   
            elif self.main_training and self.same_class_all and self.use_queue:
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                
                self.id_save_FT += 1   
            
            elif self.main_training and self.same_class:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override) 
            elif self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                
                # base novel aug
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
            else:

                if base_proposal_ious is not None:

                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                else:

                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            bbox_results['cont_feat'], 
                            gt_base_bbox[0], 
                            bbox_targets[0],
                            classes_eq, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            aug_bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            aug_bbox_targets[0], 
                            None, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
        
            if self.contrast_loss_bbox is not None:
                pos_inds_aug = (aug_bbox_targets[0] >= 0) & (aug_bbox_targets[0] > self.num_classes) 
                pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] > self.num_classes)
                base_pos_inds = (gt_base_bbox[0] >= 0) & (gt_base_bbox[0] < self.num_classes)
                
                
                # do not perform bounding box regression for BG anymore.
                
                if pos_inds.any() and pos_inds_aug.any():
                    # novel
                    if self.reg_class_agnostic:
                        bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                        labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
                    else:
                        bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                        labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], -1, 4)
                        pos_bbox_pred = pos_bbox_pred[pos_inds.type(torch.bool),labels_true]

                    # aug
                    if self.reg_class_agnostic:
                        bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                        labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
                    else:
                        bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                        labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], -1,
                            4)[pos_inds_aug.type(torch.bool),labels_true_aug]
                    # base
                    if self.reg_class_agnostic:
                        bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                        base_pos_bbox_pred = gt_base_bbox[2].view(
                            gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                    else:
                        bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                        base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                        #base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], -1, 4)[base_pos_inds.type(torch.bool),bbox_label_base]

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
                    base_pos_bbox_pred, 
                    bbox_label_base,
                    classes_eq,
                    transform_applied,
                    gt_labels_aug,
                    min_size,
                    base_bbox_pred,
                    gt_base_bbox,
                    reduction_override=reduction_override)
        
            
            if 'loss_cosine' in losses.keys():
                if losses['loss_cosine'] == -1:
                    print(f'no loss')
                    losses = dict()
        return losses
   
    @force_fp32(apply_to=('cont_feat'))
    def loss_contrast_cls(self,
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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}
            
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
        else:
            if self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                
                losses['loss_contrastive_cls'], pred_cont = self.contrast_loss_classif(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    queue_res, 
                    queue_trg, 
                    queue_iou,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    reduction_override=reduction_override)
                    
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
        
        return losses
 
@HEADS.register_module()
class Agnostic_QueueAugContrastiveBBoxHead_Branch_classqueue_replace_withbg(Shared2FCBBoxHead):
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
                 mlp_head_channels: int = 128,
                 with_weight_decay = False,
                 loss_cosine = None,
                 loss_c_cls = None,
                 loss_base_aug = None,
                 loss_c_bbox = None,
                 loss_all = None,
                 contrast_loss_classif = None,
                 to_norm_cls = False,
                 main_training = False,
                 same_class = False,
                 same_class_all = False,
                 queue_path = 'init_queue.p',
                 init_queue = False,
                 use_queue = False,
                 use_base_queue = True,
                 use_novel_queue = True,
                 use_aug_queue = True,
                 queue_length = 60,
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
            
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        if loss_all is None:
            if loss_c_cls is not None:
                self.contrast_loss_cls = build_loss(loss_c_cls)
            else:
                self.contrast_loss_cls = None

            if loss_cosine is not None:
                self.contrast_loss = build_loss(loss_cosine)
            else:
                self.contrast_loss = None
            if loss_base_aug is not None:
                self.contrast_loss_base_aug = build_loss(loss_base_aug)
            else:
                self.contrast_loss_base_aug = None
            if contrast_loss_classif is not None:
                self.contrast_loss_classif = build_loss(contrast_loss_classif)
            else:
                self.contrast_loss_classif = None
            self.contrast_loss_all = None
        else:
            self.contrast_loss_all = build_loss(loss_all)
            self.contrast_loss_cls = None
            self.contrast_loss = None
            self.contrast_loss_base_aug = None


        if loss_c_bbox is not None:
            self.contrast_loss_bbox = build_loss(loss_c_bbox)
        else:
            self.contrast_loss_bbox = None
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.queue_length = queue_length
        if self.use_queue and not init_queue:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = self.load_queue_class(use_base_queue, use_novel_queue, use_aug_queue)

            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', queue_base_ptr)
        else:
            self.queue_res = None

        self.to_norm = to_norm_cls


        self.same_class_all = same_class_all
        if self.same_class_all:
            self.same_class = False
        else: 
            self.same_class = same_class
        self.main_training = main_training
        
        self.id_save_FT = 1
        self.init_queue = init_queue
        if init_queue:
            self.queue_new_res = torch.empty((self.queue_length, mlp_head_channels))
            self.queue_new_trg = -2*torch.ones((self.queue_length))
            self.queue_new_iou = torch.ones((self.queue_length))


    def queue_init(self, new_dict):
        if self.queue_res is None:
            self.queue_res = new_dict
        else:
            z = self.queue_res.copy()   
            z.update(new_dict)
            self.queue_res = z

        with open(self.queue_path, 'wb') as fp:
            pickle.dump(self.queue_res, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @torch.no_grad()
    def update_queue_class(self, new_base, new_novel, new_aug):
        
        len_id = 0
        if new_base is not None:
            bs = new_base[0].shape[0] if len(new_base[0].shape) > 2 else 1
            fs = new_base[0].shape[1]
            cur_device = new_base[0].device
        elif new_novel is not None:
            bs = new_novel[0].shape[0] if len(new_novel[0].shape) > 2 else 1
            fs = new_novel[0].shape[1]
            cur_device = new_novel[0].device
        elif new_aug is not None:
            bs = new_aug[0].shape[0] if len(new_aug[0].shape) > 2 else 1
            fs = new_aug[0].shape[1]
            cur_device = new_aug[0].device
        else:
            assert True, " One of base, novel or aug should not be none"

        if new_base is not None:
            len_id += 1
        if new_novel is not None:
            len_id += 1
        if new_aug is not None:
            len_id += 1
        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        if queue_base_trg is not None:
            update = True
            ptr = self.queue_ptr.int() 
            reset = False
            increase = False

            for class_id in range(self.num_classes+1):
                tmp = None
                if new_base is not None:
                    if class_id in new_base[1].unique():

                        if ptr[class_id] == (self.queue_length//(self.num_classes+1)):
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -2:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_base[0][new_base[1][new_base[1] == class_id]]
                            increase = True
                            
                if new_novel is not None:

                    if class_id in new_novel[1].unique():
                        if tmp is not None:
                            tmp2 = new_novel[0][new_novel[1][new_novel[1] == class_id]]

                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//(self.num_classes+1)):
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -2:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_novel[0][new_novel[1][new_novel[1] == class_id]]
                            increase = True



                if new_aug is not None:
                    if class_id in new_aug[1].unique():
                        if tmp is not None:
                            tmp2 = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            tmp = torch.vstack([tmp, tmp2])
                        elif ptr[class_id] == (self.queue_length//(self.num_classes+1)):
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            reset = True
                        elif queue_base_trg[class_id+(self.num_classes*ptr[class_id])] == -2:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True
                        else:
                            tmp = new_aug[0][new_aug[1][new_aug[1] == class_id]]
                            increase = True

                if tmp is not None:
                    if reset:
                        ptr[class_id] = 0
                    tmp = tmp.mean(dim=0)
                    queue_base_res[class_id+((self.num_classes+1)*ptr[class_id])] = tmp
                    queue_base_trg[class_id+((self.num_classes+1)*ptr[class_id])] = class_id

                    if reset and increase:
                        assert True, "problem, reset and increase"
                    elif increase:
                        ptr[class_id] += 1
            self.queue_res = queue_base_res.to(cur_device).detach()
            self.queue_trg = queue_base_trg.to(cur_device).detach()
            self.queue_ptr = ptr
            self.queue_iou = queue_base_iou.to(cur_device).detach()
            
        else:                
            shape_queue = (self.queue_length//(self.num_classes+1))*(self.num_classes+1)
            queue_new_res = torch.empty((shape_queue, fs))
            queue_new_trg = -2*torch.ones((shape_queue))
            queue_new_iou = torch.ones((shape_queue))
            len_id = 0
            if new_base is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_base[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_novel is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_novel[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_novel[0][new_novel[1][new_novel[1] == class_id]].mean(dim=0)
                len_id += 1
            if new_aug is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_aug[1].unique():
                        queue_new_trg[class_id] = class_id
                        queue_new_res[class_id] = new_aug[0][new_aug[1][new_aug[1] == class_id]].mean(dim=0)


            
            self.queue_res = queue_new_res.to(cur_device).detach()
            self.queue_trg = queue_new_trg.to(cur_device).detach()
            self.queue_iou = queue_new_iou.to(cur_device).detach()
    
    def load_queue_class_new(self, base=True, novel=True, aug=True):
        
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
            bg_base = -1
            
            
            if 'bg' in data.keys():
                bg_base = data['bg']['trg'].item()
            for k in data.keys():
                key_type = type(k)
                break
            
            
            for key_class in range(len(data.keys())):#self.num_classes):
                if bg_base == key_class:
                    queue_res.append(data['bg'])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                else:
                    if key_type is str:
                        if type(data[str(key_class)]) is Dict:
                            queue_res.append(data[str(key_class)]['results'])
                        else:
                            queue_res.append(data[str(key_class)])
                    else:
                        if type(data[key_class]) is Dict:
                            queue_res.append(data[key_class]['results'])
                        else:
                            queue_res.append(data[key_class])
                    queue_trg.append(key_class)
                    queue_iou.append(1.)
                queue_ptr.append(1.)
            queue_new_ptr = torch.tensor(queue_ptr).long()
            if (len(data.keys())) < (self.num_classes+1):
                queue_base_ptr = torch.zeros((self.num_classes+1), dtype=torch.long)
                queue_base_ptr[:queue_new_ptr.shape[0]] = queue_new_ptr.clone()
            else:
                queue_base_ptr = queue_new_ptr.clone()

            queue_new_res = torch.stack(queue_res)
            queue_new_trg = torch.tensor(queue_trg)
            queue_new_iou = torch.tensor(queue_iou)

            fs = queue_new_res.shape[1]
            shape_queue = (self.queue_length//(self.num_classes+1))*(self.num_classes+1)

            queue_base_res = torch.zeros((shape_queue, fs))
            queue_base_res[:queue_new_res.shape[0]] = queue_new_res.clone()

            queue_base_trg = -2*torch.ones((shape_queue))
            queue_base_trg[:queue_new_trg.shape[0]] = queue_new_trg.clone()

            queue_base_iou = torch.ones((shape_queue))
            queue_base_iou[:queue_new_iou.shape[0]] = queue_new_iou.clone()

        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes+1, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

    def load_queue_class(self, base=True, novel=True, aug=True):
        
        if exists(self.queue_path):
            with open(self.queue_path, 'rb') as fp:
                data = pickle.load(fp)
            queue_res, queue_trg, queue_iou, queue_ptr = [], [], [], []
            
            for key_class in range(self.num_classes+1):
                queue_res.append(data[str(key_class)])
                queue_trg.append(key_class)
                queue_iou.append(1.)
                queue_ptr.append(1.)
            queue_new_ptr = torch.tensor(queue_ptr).long()
            queue_base_ptr = queue_new_ptr.clone()

            queue_new_res = torch.stack(queue_res)
            queue_new_trg = torch.tensor(queue_trg)
            queue_new_iou = torch.tensor(queue_iou)

            fs = queue_new_res.shape[1]
            shape_queue = (self.queue_length//(self.num_classes+1))*(1+self.num_classes)

            queue_base_res = torch.zeros((shape_queue, fs))
            queue_base_res[:queue_new_res.shape[0]] = queue_new_res.clone()

            queue_base_trg = -2*torch.ones((shape_queue))
            queue_base_trg[:queue_new_trg.shape[0]] = queue_new_trg.clone()

            queue_base_iou = torch.ones((shape_queue))
            queue_base_iou[:queue_new_iou.shape[0]] = queue_new_iou.clone()

        else:
            queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr = torch.zeros(1), None, torch.zeros(1), torch.zeros(self.num_classes+1, dtype=torch.long)
        return queue_base_res, queue_base_trg, queue_base_iou, queue_base_ptr

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
            bbox_pred, contrast_feat  = self.forward_one_branch(x, True)
            aug_bbox_pred, aug_contrast_feat = self.forward_one_branch(x_aug, True)
        else:
            bbox_pred, aug_bbox_pred = None, None
            aug_contrast_feat, contrast_feat = None, None
            
        base_bbox_pred, base_contrast_feat = self.forward_one_branch(base_x)
        
        return base_bbox_pred, base_contrast_feat, bbox_pred, contrast_feat, aug_bbox_pred, aug_contrast_feat


    def forward_one_branch(self, x, base=True):
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
        x_reg = x
        x_contra = x
       
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # contrastive branch

        cont_feat = self.contrastive_head(x_contra)
        cont_feat = F.normalize(cont_feat, dim=1)
        
        return bbox_pred, cont_feat


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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}

            if new_base is not None:
                for class_id in range(self.num_classes+1):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -2:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -2 in self.queue_new_trg.unique(), "queue init is over"
            losses = dict()
        else:
            if self.main_training and not self.same_class and not self.use_queue:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
            elif self.main_training and self.same_class and self.use_queue:
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = None, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1   
            elif self.main_training and self.same_class_all and self.use_queue:
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:

                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                
                self.id_save_FT += 1   
            
            elif self.main_training and self.same_class:
                if self.with_weight_decay:
                    decay_rate = self._decay_rate
                if self.contrast_loss_base_aug is not None:

                    losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)   
                if self.contrast_loss_cls is not None:
                    losses['loss_c_cls'] = self.contrast_loss_cls(
                        bbox_results['cont_feat'], 
                        bbox_targets[0], 
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss is not None:
                    losses['loss_cosine'] = self.contrast_loss(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        classes_equi = classes_eq, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override)
                if self.contrast_loss_all is not None:
                    losses['loss_contrastive'] = self.contrast_loss_all(
                        base_bbox_pred['cont_feat'], 
                        gt_base_bbox[0], 
                        base_proposal_ious,
                        bbox_results['cont_feat'], 
                        bbox_targets[0],
                        proposal_ious,
                        aug_bbox_results['cont_feat'], 
                        aug_bbox_targets[0], 
                        aug_proposal_ious,
                        queue_res, 
                        queue_trg, 
                        queue_iou,
                        classes_equi = None, 
                        nbr_classes = num_classes,
                        decay_rate=decay_rate,
                        reduction_override=reduction_override) 
            elif self.queue_res is not None and self.use_queue:
                #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
                
                # base novel aug
                if self.queue_trg is None:
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                else:
                    queue_res = self.queue_res.detach()
                    queue_trg = self.queue_trg.detach()
                    queue_iou = self.queue_iou.detach()
                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:

                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], # cont_feat
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    
                
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
            else:

                if base_proposal_ious is not None:

                    if self.with_weight_decay:
                        decay_rate = self._decay_rate
                    if self.contrast_loss_base_aug is not None:
                        losses['loss_c_augbase'] = self.contrast_loss_base_aug(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)   
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
                else:

                    if self.contrast_loss is not None:
                        losses['loss_cosine'] = self.contrast_loss(
                            base_bbox_pred['cont_feat'], 
                            bbox_results['cont_feat'], 
                            gt_base_bbox[0], 
                            bbox_targets[0],
                            classes_eq, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_cls is not None:
                        losses['loss_c_cls'] = self.contrast_loss_cls(
                            bbox_results['cont_feat'], 
                            aug_bbox_results['cont_feat'], 
                            bbox_targets[0], 
                            aug_bbox_targets[0], 
                            None, 
                            num_classes,
                            reduction_override=reduction_override)
                    if self.contrast_loss_all is not None:
                        losses['loss_contrastive'] = self.contrast_loss_all(
                            base_bbox_pred['cont_feat'], 
                            gt_base_bbox[0], 
                            base_proposal_ious,
                            bbox_results['cont_feat'], 
                            bbox_targets[0],
                            proposal_ious,
                            aug_bbox_results['cont_feat'], 
                            aug_bbox_targets[0], 
                            aug_proposal_ious,
                            queue_res, 
                            queue_trg, 
                            queue_iou,
                            classes_equi = classes_eq, 
                            nbr_classes = num_classes,
                            decay_rate=decay_rate,
                            reduction_override=reduction_override)
        
            if self.contrast_loss_bbox is not None:
                pos_inds_aug = (aug_bbox_targets[0] >= 0) & (aug_bbox_targets[0] > self.num_classes) 
                pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] > self.num_classes)
                base_pos_inds = (gt_base_bbox[0] >= 0) & (gt_base_bbox[0] < self.num_classes)
                
                
                # do not perform bounding box regression for BG anymore.
                
                if pos_inds.any() and pos_inds_aug.any():
                    # novel
                    if self.reg_class_agnostic:
                        bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                        labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
                    else:
                        bbox_label = bbox_targets[0][pos_inds.type(torch.bool)]
                        labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                        pos_bbox_pred = bbox_score_pred.view(
                            bbox_score_pred.shape[0], -1, 4)
                        pos_bbox_pred = pos_bbox_pred[pos_inds.type(torch.bool),labels_true]

                    # aug
                    if self.reg_class_agnostic:
                        bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                        labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
                    else:
                        bbox_label_aug = aug_bbox_targets[0][pos_inds_aug.type(torch.bool)]
                        labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                        aug_pos_bbox_pred = aug_bbox_score_pred.view(
                            aug_bbox_score_pred.shape[0], -1,
                            4)[pos_inds_aug.type(torch.bool),labels_true_aug]
                    # base
                    if self.reg_class_agnostic:
                        bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                        base_pos_bbox_pred = gt_base_bbox[2].view(
                            gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                    else:
                        bbox_label_base = gt_base_bbox[0][base_pos_inds.type(torch.bool)]

                        base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], 4)[base_pos_inds.type(torch.bool)]
                        #base_pos_bbox_pred = gt_base_bbox[2].view(gt_base_bbox[2].shape[0], -1, 4)[base_pos_inds.type(torch.bool),bbox_label_base]

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
                    base_pos_bbox_pred, 
                    bbox_label_base,
                    classes_eq,
                    transform_applied,
                    gt_labels_aug,
                    min_size,
                    base_bbox_pred,
                    gt_base_bbox,
                    reduction_override=reduction_override)
        
            
            if 'loss_cosine' in losses.keys():
                if losses['loss_cosine'] == -1:
                    print(f'no loss')
                    losses = dict()
        return losses
   
    @force_fp32(apply_to=('cont_feat'))
    def loss_contrast_cls(self,
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
                    base_proposal_ious=None,
                    proposal_ious=None,
                    aug_proposal_ious=None,
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

        if self.init_queue:

            classes_eq[num_classes]=num_classes
            if (bbox_targets[0]> num_classes).any():
                bbox_save = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cont_feat'].device)

            new_base = [base_bbox_pred['cont_feat'], bbox_save, base_proposal_ious]

            id_dict = {}
            if new_base is not None:
                for class_id in range(self.num_classes):
                    if class_id in new_base[1].unique():
                        if self.queue_new_trg[class_id] == -1:
                            self.queue_new_trg[class_id] = class_id
                            self.queue_new_res[class_id] = new_base[0][new_base[1][new_base[1] == class_id]].mean(dim=0)
                            new_dict = {str(class_id):self.queue_new_res[class_id]}
                            id_dict.update(new_dict)
            
            self.queue_init(id_dict)
            assert -1 in self.queue_new_trg.unique(), "queue init is over"
        else:
            if self.queue_res is not None and self.use_queue:

                queue_res = self.queue_res.detach()
                queue_trg = self.queue_trg.detach()
                queue_iou = self.queue_iou.detach()
                
                losses['loss_contrastive_cls'], pred_cont = self.contrast_loss_classif(
                    bbox_results['cont_feat'], 
                    bbox_targets[0], 
                    proposal_ious,
                    queue_res, 
                    queue_trg, 
                    queue_iou,
                    classes_equi = classes_eq, 
                    nbr_classes = num_classes,
                    reduction_override=reduction_override)
                    
                classes_eq[num_classes]=num_classes
                if self.use_base_queue:
                    if (gt_base_bbox[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in gt_base_bbox[0].tolist()])
                    else:
                        queue_trg = gt_base_bbox[0]
                    base_update = [base_bbox_pred['cont_feat'], queue_trg, base_proposal_ious]
                else:
                    base_update = None
                if self.use_novel_queue:
                    if (bbox_targets[0] > num_classes).any():                    
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0].tolist()])
                    else:
                        queue_trg = bbox_targets[0]
                    novel_update = [bbox_results['cont_feat'], queue_trg, proposal_ious]
                else:
                    novel_update = None

                if self.use_aug_queue:
                    if (aug_bbox_targets[0] > num_classes).any():
                        queue_trg = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0].tolist()])
                    else:
                        queue_trg = aug_bbox_targets[0]
                    aug_update = [aug_bbox_results['cont_feat'], queue_trg, aug_proposal_ious]
                else:
                    aug_update = None

                self.update_queue_class(base_update, novel_update, aug_update )
                self.id_save_FT += 1          
        
        return losses
 
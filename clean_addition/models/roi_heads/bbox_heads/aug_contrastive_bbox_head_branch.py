# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from .convfc_bbox_head import Shared2FCBBoxHeadUpdate
import numpy as np
import pickle 
@HEADS.register_module()
class AugContrastiveBBoxHead_Branch(Shared2FCBBoxHeadUpdate):
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
class QueueAugContrastiveBBoxHead_Branch(Shared2FCBBoxHeadUpdate):
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
                 main_training = False,
                 queue_path = 'init_queue.p',
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
        
        self.queue_path = queue_path
        self.use_queue = use_queue

        self.use_base_queue = use_base_queue
        self.use_novel_queue = use_novel_queue
        self.use_aug_queue = use_aug_queue

        self.queue_length = queue_length
        if self.use_queue:
            queue_base_res, queue_base_trg, queue_base_iou = self.load_queue_2(use_base_queue, use_novel_queue, use_aug_queue)
            self.register_buffer('queue_res', queue_base_res)
            self.register_buffer('queue_trg', queue_base_trg)
            self.register_buffer('queue_iou', queue_base_iou)
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        else:
            self.queue_res = None

        self.to_norm = to_norm_cls

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

        queue_base_res = self.queue_res
        queue_base_trg = self.queue_trg
        queue_base_iou = self.queue_iou
        
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
        queue_length_frac = int(queue_base_res.shape[0]/len_id)

        if queue_base_res.shape[0] < self.queue_length:
            
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
                queue_new_res += [queue_base_res[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[0], dim=0)]
                queue_new_trg += [queue_base_trg[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[1], dim=0)]
                queue_new_iou += [queue_base_iou[len_id * queue_length_frac:(len_id + 1) * queue_length_frac], torch.unsqueeze(new_novel[2], dim=0)]
                len_id += 1
            if new_aug is not None:
                queue_new_res += [queue_base_res[len_id * queue_length_frac:], torch.unsqueeze(new_aug[0], dim=0)]
                queue_new_trg += [queue_base_trg[len_id * queue_length_frac:], torch.unsqueeze(new_aug[1], dim=0)]
                queue_new_iou += [queue_base_iou[len_id * queue_length_frac:], torch.unsqueeze(new_aug[2], dim=0)]

            queue_new_res = torch.cat(queue_new_res) 
            queue_new_trg = torch.cat(queue_new_trg) 
            queue_new_iou = torch.cat(queue_new_iou) 

            self.queue_res = queue_new_res.detach()
            self.queue_trg = queue_new_trg.detach()
            self.queue_iou = queue_new_iou.detach()

        else:
            update = True
            ptr = int(self.queue_ptr) 
            len_id = 0
            if new_base is not None:
                self.queue_res[ptr,:] = new_base[0].detach()
                self.queue_trg[ptr,:] = new_base[1].detach()
                self.queue_iou[ptr,:] = new_base[2].detach()
                len_id += 1
            queue_shape = self.queue_trg[len_id * queue_length_frac+ptr,:].shape[0]
            if new_novel is not None:
                
                if (new_novel[0].shape[0] != queue_shape) or (new_novel[1].shape[0] != queue_shape) or (new_novel[2].shape[0] != queue_shape):
                    print(f'new novel {new_novel[0].shape}')
                    print(f'new novel 1 {new_novel[1].shape}')
                    print(f'new novel 2 {new_novel[2].shape}')

                    print(f'queue trg {self.queue_trg[len_id * queue_length_frac+ptr,:].shape}')
                    print(f'queue res {self.queue_res[len_id * queue_length_frac+ptr,:].shape}')
                    print(f'queue iou {self.queue_iou[len_id * queue_length_frac+ptr,:].shape}')
                    update = False
                else:
                    self.queue_res[len_id * queue_length_frac+ptr,:] = new_novel[0].detach()
                    self.queue_trg[len_id * queue_length_frac+ptr,:] = new_novel[1].detach()
                    self.queue_iou[len_id * queue_length_frac+ptr,:] = new_novel[2].detach()
                    len_id += 1
            if new_aug is not None:
                if (new_aug[0].shape[0] != queue_shape) or (new_aug[1].shape[0] != queue_shape) or (new_aug[2].shape[0] != queue_shape):
                    print(f'new novel {new_aug[0].shape}')
                    print(f'new novel 1 {new_aug[1].shape}')
                    print(f'new novel 2 {new_aug[2].shape}')

                    print(f'queue trg {self.queue_trg[len_id * queue_length_frac+ptr,:].shape}')
                    print(f'queue res {self.queue_res[len_id * queue_length_frac+ptr,:].shape}')
                    print(f'queue iou {self.queue_iou[len_id * queue_length_frac+ptr,:].shape}')
                    update = False
                else:
                    self.queue_res[len_id * queue_length_frac+ptr,:] = new_aug[0].detach()
                    self.queue_trg[len_id * queue_length_frac+ptr,:] = new_aug[1].detach()
                    self.queue_iou[len_id * queue_length_frac+ptr,:] = new_aug[2].detach()
            if update:
                ptr = (ptr + 1) % queue_length_frac  # move pointer

                self.queue_ptr[0] = ptr

    def load_queue(self, base=True, novel=True, aug=True):
        print('init load')
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
        print('init load')
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


        print(f'queue shape {len(queue_trg)}')
        queue_base_res = torch.stack(queue_res)
        queue_base_trg = torch.stack(queue_trg)
        queue_base_iou = torch.stack(queue_iou)

        print(f'queue shape {queue_base_trg.shape}')

        id_shuffle = np.arange(queue_base_res.shape[0])
        np.random.shuffle(id_shuffle)
        queue_base_res = queue_base_res[id_shuffle]
        queue_base_trg = queue_base_trg[id_shuffle]
        queue_base_iou = queue_base_iou[id_shuffle]
        
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
        if self.main_training:
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

        elif self.queue_res is not None and self.use_queue:
            #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg, queue_base_iou, queue_novel_iou, queue_aug_iou = self.queue
            queue_res = self.queue_res.detach()
            queue_trg = self.queue_trg.detach()
            queue_iou = self.queue_iou.detach()
            # base novel aug
            if queue_trg is None:
                print('yes')
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
            
            if self.use_base_queue:
                base_update = [base_bbox_pred['cont_feat'], gt_base_bbox[0], base_proposal_ious]
            else:
                base_update = None

            if self.use_novel_queue:
                novel_update = [bbox_results['cont_feat'], bbox_targets[0], proposal_ious]
            else:
                novel_update = None

            if self.use_aug_queue:
                aug_update = [aug_bbox_results['cont_feat'], aug_bbox_targets[0], aug_proposal_ious]
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
        
        return losses
# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models import weight_reduce_loss

@LOSSES.register_module()
class CosineSim(nn.Module):

    def __init__(self,
                 margin = .5,
                 max_contrastive_loss = 3.,
                 loss_weight=.5):
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
        super(CosineSim, self).__init__()
        
        self.margin = margin
        self.loss_weight = loss_weight
        self.max_contrastive_loss = max_contrastive_loss

    def forward(self,
                base_bbox_pred,
                cls_score_pred,
                gt_base_bbox,
                bbox_targets,
                classes_eq,
                num_classes,
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
        base_cls = base_bbox_pred
        novel_cls = cls_score_pred
        
        pos_inds = (bbox_targets >= 0) & (bbox_targets > num_classes)
        if classes_eq is not None:
            base_pos_inds = (gt_base_bbox >= 0) & (gt_base_bbox < num_classes)
        else:
            base_pos_inds = (gt_base_bbox >= 0) & (gt_base_bbox > num_classes)
            
        base_trg = gt_base_bbox[base_pos_inds.type(torch.bool)]
        novel_trg = bbox_targets[pos_inds.type(torch.bool)]

        base_cls = base_cls[base_pos_inds.type(torch.bool)]
        novel_cls = novel_cls[pos_inds.type(torch.bool)]


        dot_mat = torch.mm(base_cls,novel_cls.T)

        n_x1 = torch.unsqueeze(torch.sqrt(torch.sum(base_cls*base_cls,dim=1)),dim=1)
        n_x2 = torch.unsqueeze(torch.sqrt(torch.sum(novel_cls*novel_cls,dim=1)),dim=0)

        len_mat = torch.clip(n_x1*n_x2, min=1e-08)

        cos_sim_mat = dot_mat / len_mat
        if classes_eq is not None:
            novel_trg = torch.tensor([classes_eq[int(i)] for i in novel_trg])

        cos_mask = (torch.unsqueeze(novel_trg.to(base_trg.get_device()), dim=0)==(torch.unsqueeze(base_trg, dim=1)))
        inv_mask = ~cos_mask
        cos_mask = cos_mask.type(torch.int32)
        inv_mask = inv_mask.type(torch.int32)

        loss_cos = ((cos_mask-cos_sim_mat*cos_mask)+(torch.clip((cos_sim_mat*inv_mask-self.margin*inv_mask), min=0))).mean()

        return torch.clip(self.loss_weight*loss_cos, max=self.max_contrastive_loss)

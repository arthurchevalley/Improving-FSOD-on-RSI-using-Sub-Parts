import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet.core import bbox_overlaps
from mmdet.models.builder import LOSSES

import cv2
import random

import copy
import numpy as np
import mmcv
from mmdet.models.losses.utils import weighted_loss, weight_reduce_loss



@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
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
    if torch.isnan(target).any():
        print('nan trg')
        target[torch.isnan(target)] = 0.0

    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    
    return loss

@LOSSES.register_module()
class ConstellationLossCLS(nn.Module):

    def __init__(self,
                 K = 4,
                 max_contrastive_loss = 1,
                 loss_weight=1.0,
                 reduction = 'mean',
                 activated=False):
        """
            Tests of a constellation loss for classification. Adapted from https://arxiv.org/abs/1905.10675
            Leads to worst results than the supervised contrastive loss
        """
        super(ConstellationLossCLS, self).__init__()
        self.K = K
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.max_contrastive_loss = max_contrastive_loss

    def forward(self,
                label_targets,
                label_targets_aug,
                label_targets_aug_true,
                aug_cls_score_pred,
                cls_score_pred, 
                bbox_targets,
                aug_bbox_targets,
                transfo,

                weight=None,
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


        """Build the constellation loss over a batch of embeddings.
        Args:
            target: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            ctl_loss: scalar tensor containing the constellation loss
        """

        # cls 
        
        BATCH_SIZE = len(label_targets_aug)

        # cls_score_pred size: sampled number x classes
        # cls_score_pred size: sampled number x classes
        
        true_aug_labels = []
        for i in range(BATCH_SIZE):
            true_aug_labels += label_targets_aug[i].tolist()

        labels = bbox_targets[0].tolist()
        aug_labels = aug_bbox_targets[0].tolist()
        loss_list = []

        for i in range(len(aug_labels)): 
            if aug_labels[i] in true_aug_labels:
                f_i_a = aug_cls_score_pred[i,:]
                f_j_n = []
                f_i_p = []
                for j in range(len(labels)):
                    if (labels[j] != aug_labels[i]) and (labels[j] in true_aug_labels):
                        f_j_n.append(cls_score_pred[j,:])
                    elif labels[j] == aug_labels[i]:
                        f_i_p.append(cls_score_pred[j,:])
                if len(f_j_n) and len(f_i_p):
                    f_j_n = torch.stack(random.sample(f_j_n, min(self.K, len(f_j_n))))
                    f_i_p = torch.stack(random.sample(f_i_p, 1))
                else:
                    continue

                f_i_p = f_i_p.expand(f_j_n.shape[0], f_j_n.shape[1])
                anchor_negative_dist = torch.matmul(f_i_a, f_j_n.T)
                anchor_positive_dist = torch.matmul(f_i_a, f_i_p.T)
                
                dist = anchor_negative_dist - anchor_positive_dist
                loss_list.append(dist)
        
        if len(loss_list):
            ctl_loss = 0
            for i in range(len(loss_list)):

                ctl_loss += torch.log(torch.sum(torch.exp(loss_list[i])) + 1.)
                
            ctl_loss /= len(loss_list) 
            ctl_loss = self.loss_weight*torch.clip(ctl_loss, min=0, max=self.max_contrastive_loss)
        else:
            ctl_loss = torch.ones(1)*self.max_contrastive_loss
        return ctl_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ciou_loss(pred, target, eps=1e-7):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])

    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    
    
    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v + eps)
    
    do_print = False
    if do_print:
        print(f'enclose_x1y1 {enclose_x1y1}')
        print(f'enclose_x2y2 {enclose_x2y2}')
        print(f'cw {cw}')
        print(f'ch {ch}')
        print(f'w1 h1  {w1, h1}')
        print(f'w2 h2  {w2, h2}')
        print(f'left {left}')
        print(f'right {right}')

        print(f'rho2 {rho2}')
        print(f'factor {factor}')
        print(f'v {v}')
        print(f'iou {ious}')
        print(f'alpha {alpha}')
        print(f'lt {lt}')
        print(f'rb {rb}')
        print(f'wh {wh}')
        print(f'ap {ap}')
        print(f'ag {ag}')

        print(f'overlap {overlap}')
        print(f'union {union}')
    # CIoU
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss

@LOSSES.register_module()
class IoULossBBOX(nn.Module):

    """
        Tests of various IoU loss for regression. 
        No good results achieved
    """
    def __init__(self,
                 eps=1e-6,
                 K = 4,
                 max_contrastive_loss = 1,
                 loss_weight=1.0,
                 reduction = 'mean',
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
        super(IoULossBBOX, self).__init__()
        self.K = K
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.max_contrastive_loss = max_contrastive_loss

    def forward(self,
                label_targets,
                label_targets_aug,
                label_targets_aug_true,
                label_targets_true,
                aug_bbox_pred,
                bbox_pred, 
                bbox_targets,
                aug_bbox_targets,
                img_metas,
                gt_labels_aug,
                min_size,
                base_bbox_pred,
                gt_base_bbox,
                weight=None,
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

        Build the constellation loss over a batch of embeddings.
        Args:
            target: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            ctl_loss: scalar tensor containing the constellation loss
        """

        loss_list = []

        batch_size = len(gt_labels_aug)
        bbox_per_batch = []
        for i in range(batch_size):
            bbox_per_batch.append(gt_labels_aug[i].shape[0])
            if i:
                bbox_per_batch[i] += bbox_per_batch[i-1]
        bbox_per_batch = torch.tensor(bbox_per_batch)
        bbox_in_batch = []
        for i in range(batch_size):
            bbox_in_batch += gt_labels_aug[i].tolist()

        transform_applied = []
        img_shape = []
        for i in range(len(img_metas)):
            if 'applied_transformation' in img_metas[i]:
                transform_applied.append(img_metas[i]['applied_transformation'])
                img_shape.append(img_metas[i]['img_shape'])
            else:
                transform_applied.append(['vanilla'])

        current_batch_id = 0
        batch_cntr = 0
        length_of_batch = bbox_per_batch[current_batch_id]
        aug_seen_id = []
        mask = []
        aug_mask = []
        
        for i in range(label_targets_aug.shape[0]): 
            if label_targets_aug[i] in label_targets.unique():
                aug_mask.append(True)
                aug_seen_id.append(label_targets_aug[i])
                
                value_check = torch.tensor([bbox_in_batch.index(label_targets_aug[i]) for j in range(len(bbox_per_batch))])
                current_batch_id = bbox_per_batch.tolist().index(bbox_per_batch[value_check < bbox_per_batch][0]) 
                
                f_i_a = aug_bbox_pred[i,:]
                flipped = f_i_a.clone()
                w = 800
                h = 800
                for transfo in transform_applied[current_batch_id]:  
                    if transfo != 'horizontal' and transfo != 'vertical' and transfo != 'diagonal' and transfo != 'vanilla':
                        #print(f'transformation {transfo}')   
                        if transfo == 'horizontal':
                            w = 800#img_shape[current_batch_id][0]
                            flipped[0::4] = w - f_i_a[2::4]
                            flipped[2::4] = w - f_i_a[0::4]
                        elif transfo == 'vertical':
                            h = 800#img_shape[current_batch_id][1]
                            flipped[1::4] = h - f_i_a[3::4]
                            flipped[3::4] = h - f_i_a[1::4]
                        elif transfo == 'diagonal':
                            w = 800#img_shape[current_batch_id][0]
                            h = 800#img_shape[current_batch_id][1]
                            flipped[0::4] = w - f_i_a[2::4]
                            flipped[1::4] = h - f_i_a[3::4]
                            flipped[2::4] = w - f_i_a[0::4]
                            flipped[3::4] = h - f_i_a[1::4]
                        elif transfo == '-90' or transfo == '90' or transfo == '-180' or transfo == '180':
                            angle = -int(transfo)
                            if angle > 0:
                                # CCW
                                xmin, ymin, xmax, ymax = f_i_a[1], w-f_i_a[2], f_i_a[3], w-f_i_a[0]
                                
                                if angle > 90:
                                    xmin, ymin, xmax, ymax = ymin, w-xmax, ymax, w-xmin
                            else:
                                #CW
                                xmin, ymin, xmax, ymax = h-f_i_a[3], f_i_a[0], h-f_i_a[1], f_i_a[2]
                                if angle < -90:
                                    xmin, ymin, xmax, ymax = h-ymax, xmin, h-ymin, xmax
                            flipped = torch.tensor([xmin, ymin, xmax, ymax])  
                            
                    else:
                        flipped = f_i_a.clone()

                    f_i_a = flipped.clone()
                aug_bbox_pred[i,:] = f_i_a.clone()
                
            else:
                aug_mask.append(False)
                continue

        seen_id = []
        for i in range(label_targets.shape[0]): 
            if label_targets[i] in label_targets_aug.unique(): #and label_targets[i] not in seen_id:
                mask.append(True)
                seen_id.append(label_targets[i])
            else:
                mask.append(False)

        
        bbox_pred = bbox_pred[mask]
        label_targets = label_targets[mask]
        aug_bbox_pred = aug_bbox_pred[aug_mask]
        label_targets_aug = label_targets_aug[aug_mask]


        final_aug = []
        final = []
        
        for i in label_targets.unique().tolist():

            lbl_id = label_targets == i
            aug_lbl_id = label_targets_aug == i
            bb1 = aug_bbox_pred[aug_lbl_id]
            bb2 = bbox_pred[lbl_id] 
            if bb1.shape[0] >= bb2.shape[0]:
                kept_aug_bbox, kept_bbox = self.compute_iou(bb1, bb2)
            else:
                kept_bbox, kept_aug_bbox = self.compute_iou(bb2, bb1)
            final_aug.append(kept_aug_bbox.to(aug_bbox_pred.get_device()))
            final.append(kept_bbox.to(bbox_pred.get_device()))
        final = torch.stack(final).to(bbox_pred.get_device())
        final_aug = torch.stack(final_aug).to(aug_bbox_pred.get_device())
        #print(f'final {final}')
        #print(f'final aug {final_aug}')
        bbox_loss = self.loss_weight * ciou_loss(
            final_aug,
            final,
            None,
            eps=self.eps,
            reduction='mean',
            avg_factor=None)
            
        return bbox_loss

    def compute_iou(self,bb1, bb2, min_area=5):
        
        gious =  bbox_overlaps(bb1, bb2, mode='giou')
        id = gious.argmax()
        bb1_id = id%bb1.shape[0]
        bb2_id = torch.div(id, bb1.shape[0], rounding_mode='floor')
        bb1_to_keep = bb1[bb1_id]
        bb2_to_keep = bb2[bb2_id]


        return bb1_to_keep, bb2_to_keep
    
@LOSSES.register_module()
class ConstellationLossBBOX(nn.Module):
    

    def __init__(self,
                 beta=1./9.,
                 K = 4,
                 max_contrastive_loss = 1,
                 loss_weight=1.0,
                 reduction = 'mean',
                 activated=False):
        """
            Tests of a constellation loss for regression. Adapted from https://arxiv.org/abs/1905.10675
            No results from it
        """
        super(ConstellationLossBBOX, self).__init__()
        self.K = K
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.max_contrastive_loss = max_contrastive_loss

    def forward(self,
                label_targets,
                label_targets_aug,
                label_targets_aug_true,
                label_targets_true,
                aug_bbox_pred,
                bbox_pred, 
                bbox_targets,
                aug_bbox_targets,
                img_metas,
                gt_labels_aug,
                min_size,
                base_bbox_pred,
                gt_base_bbox,
                weight=None,
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

        Build the constellation loss over a batch of embeddings.
        Args:
            target: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            ctl_loss: scalar tensor containing the constellation loss
        """

        loss_list = []

        batch_size = len(gt_labels_aug)
        bbox_per_batch = []
        for i in range(batch_size):
            bbox_per_batch.append(gt_labels_aug[i].shape[0])
            if i:
                bbox_per_batch[i] += bbox_per_batch[i-1]
        bbox_per_batch = torch.tensor(bbox_per_batch)
        bbox_in_batch = []
        for i in range(batch_size):
            bbox_in_batch += gt_labels_aug[i].tolist()

        transform_applied = []
        img_shape = []
        for i in range(len(img_metas)):
            if 'applied_transformation' in img_metas[i]:
                transform_applied.append(img_metas[i]['applied_transformation'])
                img_shape.append(img_metas[i]['img_shape'])
            else:
                transform_applied.append(['vanilla'])

        current_batch_id = 0
        batch_cntr = 0
        length_of_batch = bbox_per_batch[current_batch_id]
        aug_seen_id = []
        mask = []
        aug_mask = []
        
        for i in range(label_targets_aug.shape[0]): 
            if label_targets_aug[i] in label_targets.unique():
                aug_mask.append(True)
                aug_seen_id.append(label_targets_aug[i])
                
                value_check = torch.tensor([bbox_in_batch.index(label_targets_aug[i]) for j in range(len(bbox_per_batch))])
                current_batch_id = bbox_per_batch.tolist().index(bbox_per_batch[value_check < bbox_per_batch][0]) 
                
                f_i_a = aug_bbox_pred[i,:]
                flipped = f_i_a.clone()
                w = 800
                h = 800
                for transfo in transform_applied[current_batch_id]:  
                    if transfo != 'horizontal' and transfo != 'vertical' and transfo != 'diagonal' and transfo != 'vanilla':

                        if transfo == 'horizontal':
                            w = 800
                            flipped[0::4] = w - f_i_a[2::4]
                            flipped[2::4] = w - f_i_a[0::4]
                        elif transfo == 'vertical':
                            h = 800
                            flipped[1::4] = h - f_i_a[3::4]
                            flipped[3::4] = h - f_i_a[1::4]
                        elif transfo == 'diagonal':
                            w = 800
                            h = 800
                            flipped[0::4] = w - f_i_a[2::4]
                            flipped[1::4] = h - f_i_a[3::4]
                            flipped[2::4] = w - f_i_a[0::4]
                            flipped[3::4] = h - f_i_a[1::4]
                        elif transfo == '-90' or transfo == '90' or transfo == '-180' or transfo == '180':
                            angle = -int(transfo)
                            if angle > 0:
                                # CCW
                                xmin, ymin, xmax, ymax = f_i_a[1], w-f_i_a[2], f_i_a[3], w-f_i_a[0]
                                
                                if angle > 90:
                                    xmin, ymin, xmax, ymax = ymin, w-xmax, ymax, w-xmin
                            else:
                                #CW
                                xmin, ymin, xmax, ymax = h-f_i_a[3], f_i_a[0], h-f_i_a[1], f_i_a[2]
                                if angle < -90:
                                    xmin, ymin, xmax, ymax = h-ymax, xmin, h-ymin, xmax
                            flipped = torch.tensor([xmin, ymin, xmax, ymax])  
                            
                    else:
                        flipped = f_i_a.clone()

                    f_i_a = flipped.clone()
                aug_bbox_pred[i,:] = f_i_a.clone()
                
            else:
                aug_mask.append(False)
                continue

        seen_id = []
        for i in range(label_targets.shape[0]): 
            if label_targets[i] in label_targets_aug.unique(): 
                mask.append(True)
                seen_id.append(label_targets[i])
            else:
                mask.append(False)

        
        bbox_pred = bbox_pred[mask]
        label_targets = label_targets[mask]
        aug_bbox_pred = aug_bbox_pred[aug_mask]
        label_targets_aug = label_targets_aug[aug_mask]


        final_aug = []
        final = []
        
        for i in label_targets.unique().tolist():

            lbl_id = label_targets == i
            aug_lbl_id = label_targets_aug == i
            bb1 = aug_bbox_pred[aug_lbl_id]
            bb2 = bbox_pred[lbl_id]  
            if bb1.shape[0] >= bb2.shape[0]:
                kept_aug_bbox, kept_bbox = self.compute_iou(bb1, bb2)
            else:
                kept_bbox, kept_aug_bbox = self.compute_iou(bb2, bb1)
            final_aug.append(kept_aug_bbox.to(aug_bbox_pred.get_device()))
            final.append(kept_bbox.to(bbox_pred.get_device()))
        final = torch.stack(final).to(bbox_pred.get_device())
        final_aug = torch.stack(final_aug).to(aug_bbox_pred.get_device())

        bbox_loss = self.loss_weight * smooth_l1_loss(
            final_aug,
            final,
            beta=self.beta,
            reduction='mean',
            avg_factor=None)
            
        return bbox_loss

    def compute_iou(self,bb1, bb2, min_area=5):
        # bb1 has >= than bb2
        
        biggest_iou = -1
        bb1_to_keep = torch.zeros(4)
        bb2_to_keep = torch.zeros(4)
        for bb1_id in range(bb1.shape[0]):
            for bb2_id in range(bb2.shape[0]):
                print(f'bb1 {bb1[bb1_id]} bb2 {bb2[bb2_id]}')                

                x_left = max(bb1[bb1_id][0], bb2[bb2_id][0])
                y_top = max(bb1[bb1_id][1], bb2[bb2_id][1])
                x_right = min(bb1[bb1_id][2], bb2[bb2_id][2])
                y_bottom = min(bb1[bb1_id][3], bb2[bb2_id][3])
                
                if (x_right < x_left or y_bottom < y_top) or ():
                    iou = -1.
                else:
                    # The intersection of two axis-aligned bounding boxes is always an
                    # axis-aligned bounding box
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    if intersection_area < min_area:
                        iou = -1
                    else:
                        # compute the area of both AABBs
                        bb1_area = (bb1[bb1_id][2] - bb1[bb1_id][0]) * (bb1[bb1_id][3] - bb1[bb1_id][1])
                        bb2_area = (bb2[bb2_id][2] - bb2[bb2_id][0]) * (bb2[bb2_id][3] - bb2[bb2_id][1])
                        
                        # compute the intersection over union by taking the intersection
                        # area and dividing it by the sum of prediction + ground-truth
                        # areas - the interesection area
                        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
                if iou > biggest_iou:
                    biggest_iou = iou
                    bb1_to_keep = bb1[bb1_id]
                    bb2_to_keep = bb2[bb2_id]
        return bb1_to_keep, bb2_to_keep

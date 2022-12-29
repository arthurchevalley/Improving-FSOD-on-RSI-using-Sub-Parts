# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
from mmdet.models import BBoxHead

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms

@HEADS.register_module()
class ConvFCBBoxHeadUpdate(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 inplace_relu = True,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadUpdate, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
            
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

    
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=inplace_relu)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
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


@HEADS.register_module()
class Shared2FCBBoxHeadUpdate(ConvFCBBoxHeadUpdate):

    def __init__(self, fc_out_channels=1024, cosine_loss=None, num_cls_fcs=0, num_reg_fcs=0, *args, **kwargs):
        super(Shared2FCBBoxHeadUpdate, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=num_cls_fcs,
            num_reg_convs=0,
            num_reg_fcs=num_reg_fcs,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        if cosine_loss is not None:
            self.loss_cosine = build_loss(cosine_loss)
        else:
            self.loss_cosine = None
        

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def c_loss(self,
            x,
            gt_nbboxes,
            gt_nlabels,
            x_aug,
            bbox_results,
            aug_bbox_results,
            gt_bboxes_aug,
            gt_labels_aug,
            gt_labels_aug_true,
            transform_applied, 
            bbox_targets,
            aug_bbox_targets,
            bbox_score_pred,
            aug_bbox_score_pred,
            base_bbox_pred,
            gt_base_bbox,
            reduction_override=None):

        

        aug_cls_score_pred = aug_bbox_results['cls_score']
        #aug_bbox_score_pred = aug_bbox_results['bbox_pred']
        cls_score_pred = bbox_results['cls_score']
        #bbox_score_pred = bbox_results['bbox_pred']
        bbox_feat = bbox_results['bbox_feats']
        aug_bbox_feat = aug_bbox_results['bbox_feats']
        losses = dict()
        

        gt_nlabels_tmp = []
        gt_labels_aug_tmp = []

        for i in range(len(gt_labels_aug_true)):
            gt_labels_aug_tmp += gt_labels_aug_true[i].tolist()
            gt_nlabels_tmp += gt_nlabels[i].tolist()
        
        classes_eq = {gt_nlabels_tmp[i]: gt_labels_aug_tmp[i] for i in range(len(gt_nlabels_tmp))}
        
        # Call cosine loss here
        if self.loss_cosine is not None:
            cosine_loss = self.loss_cosine(base_bbox_pred['cls_score'], cls_score_pred, gt_base_bbox[0], bbox_targets[0],classes_eq, self.num_classes)
            if isinstance(cosine_loss, dict):
                losses.update(cosine_loss)
            else:
                losses['loss_CosineSim'] = cosine_loss


        loss_c_cls_ = self.loss_cosine(cls_score_pred, aug_cls_score_pred, bbox_targets[0], aug_bbox_targets[0], None, self.num_classes)

        #loss_c_cls_ = self.loss_cls(
        #    gt_nlabels,
        #    gt_labels_aug,
        #    gt_labels_aug_true,
        #    aug_cls_score_pred,
        #    cls_score_pred, 
        #    bbox_targets,
        #    aug_bbox_targets,
        #    transform_applied,
        #    reduction_override)
        if isinstance(loss_c_cls_, dict):
            losses.update(loss_c_cls_)
        else:
            losses['loss_c_cls'] = loss_c_cls_

                
        c_bbox_loss = True
        if c_bbox_loss:

            #bg_class_ind = self.num_classes
            

            labels_after_roi_aug = aug_bbox_targets[0]
            labels_after_roi = bbox_targets[0]

            bbox_trg_aug = aug_bbox_targets[2]
            bbox_trg = bbox_targets[2]
            pos_inds_aug = (labels_after_roi_aug >= 0) & (labels_after_roi_aug > self.num_classes) 
            pos_inds = (labels_after_roi >= 0) & (labels_after_roi > self.num_classes)
            
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any() and pos_inds_aug.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_score_pred.view(
                        bbox_score_pred.shape[0], 4)[pos_inds.type(torch.bool)]
                else:
                    bbox_label = labels_after_roi[pos_inds.type(torch.bool)]
                    labels_true = torch.tensor([classes_eq[int(i)] for i in bbox_label])

                    pos_bbox_pred = bbox_score_pred.view(
                        bbox_score_pred.shape[0], -1, 4)[pos_inds.type(torch.bool),labels_true]
                
                if self.reg_class_agnostic:
                    aug_pos_bbox_pred = aug_bbox_score_pred.view(
                        aug_bbox_score_pred.shape[0], 4)[pos_inds_aug.type(torch.bool)]
                else:
                    bbox_label_aug = labels_after_roi_aug[pos_inds_aug.type(torch.bool)]
                    labels_true_aug = torch.tensor([classes_eq[int(i)] for i in bbox_label_aug])

                    aug_pos_bbox_pred = aug_bbox_score_pred.view(
                        aug_bbox_score_pred.shape[0], -1,
                        4)[pos_inds_aug.type(torch.bool),labels_true_aug]
                
                min_size = min(aug_pos_bbox_pred.shape[0], pos_bbox_pred.shape[0])
                #aug_pos_bbox_pred = aug_pos_bbox_pred[:min_size]
                #pos_bbox_pred = pos_bbox_pred[:min_size]
                        
                loss_c_bbox_ = self.loss_bbox(
                    bbox_label,
                    bbox_label_aug,
                    labels_true_aug,
                    labels_true,
                    aug_pos_bbox_pred,
                    pos_bbox_pred, 
                    bbox_trg[pos_inds.type(torch.bool)],
                    bbox_trg_aug[pos_inds_aug.type(torch.bool)],
                    transform_applied,
                    gt_labels_aug,
                    min_size,
                    base_bbox_pred,
                    gt_base_bbox,
                    )

                if isinstance(loss_c_bbox_, dict):
                    losses.update(loss_c_bbox_)
                else:
                    losses['loss_c_bbox_'] = loss_c_bbox_

            else:
                losses['loss_c_bbox'] = bbox_pred[pos_inds].sum()
            
        return losses


@HEADS.register_module()
class Shared4Conv1FCBBoxHeadUpdate(ConvFCBBoxHeadUpdate):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHeadUpdate, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

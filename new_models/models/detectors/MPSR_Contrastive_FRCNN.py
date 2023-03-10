# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models import BaseDetector
from mmdet.models import TwoStageDetector
import mmcv
import numpy as np
from mmcv.runner import BaseModule, auto_fp16

@DETECTORS.register_module()
class MPSR_CosSimFRCNN(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 rpn_select_levels,
                 roi_select_levels, 
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 *args, 
                 **kwargs):

        super(MPSR_CosSimFRCNN, self).__init__(init_cfg=init_cfg,*args, **kwargs)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
     

    @auto_fp16(apply_to=('auxiliary_img_list', ))
    def extract_auxiliary_feat(self, auxiliary_img_list):
        """Extract and select features from data list at multiple scale.

        Args:
            auxiliary_img_list (list[Tensor]): List of data at different
                scales. In most cases, each dict contains: `img`, `img_metas`,
                `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`.

        Returns:
            tuple:
                rpn_feats (list[Tensor]): Features at multiple scale used
                    for rpn head training.
                roi_feats (list[Tensor]): Features at multiple scale used
                    for roi head training.
        """

        rpn_feats = []
        roi_feats = []
        for scale, img in enumerate(auxiliary_img_list):
            feats = self.backbone(img)
            if self.with_neck:
                feats = self.neck(feats)
            assert len(feats) >= self.num_fpn_levels, \
                f'minimum number of fpn levels is {self.num_fpn_levels}.'
            # for each scale of image, only one level of fpn features will be
            # selected for training.
            if scale == 5:
                # 13 x 13 -> 9 x 9
                rpn_feats.append(feats[self.rpn_select_levels[scale]][:, :,
                                                                      2:-2,
                                                                      2:-2])
            else:
                rpn_feats.append(feats[self.rpn_select_levels[scale]])
            roi_feats.append(feats[self.roi_select_levels[scale]])
        return rpn_feats, roi_feats

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
    
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, base=None, novel=None, augmented=None, auxiliary_data=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            if auxiliary_data is not None:
                # collect data or data info at same scale into one dict
                keys = list(auxiliary_data.keys())
                num_scales = max(map(int, [key[-1] for key in keys])) + 1
                auxiliary_data_list = [{
                    key.replace(f'_scale_{scale}', ''): auxiliary_data[key]
                    for key in keys if f'_scale_{scale}' in key
                } for scale in range(num_scales)]
                return self.forward_train(base, novel, augmented, img, img_metas, auxiliary_data_list, **kwargs)
            if base is not None:
                return self.forward_train(base, novel, augmented, img, img_metas, **kwargs)
            else:
                return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self,
                      base,
                      novel, 
                      augmented,
                      img, 
                      img_metas,
                      auxiliary_data_list,
                      transform = None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        device = img.get_device()
        img = base['img']
        gt_bboxes = base['gt_bboxes']
        gt_labels = base['gt_labels']


        aug_img = augmented['img'].to(img.get_device())
        gt_nbboxes = novel['gt_bboxes']
        gt_nlabels = novel['gt_labels']
        aug_gt_nbboxes = augmented['gt_bboxes']
        aug_gt_nlabels = augmented['gt_labels']

        gt_nlabels_true = augmented['gt_labels_true'] 

        # train model with refine pipeline
        auxiliary_img_list = [data['img'] for data in auxiliary_data_list]
        auxiliary_rpn_feats, auxiliary_roi_feats = self.extract_auxiliary_feat(auxiliary_img_list)


        # Classic Faster R-CNN
        losses = dict()

        img_features = self.extract_feat(img)
        aug_img_features = self.extract_feat(aug_img)
        

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                img_features,
                auxiliary_rpn_feats,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            
            rpn_losses, proposal_list_c = self.rpn_head.forward_train(
                aug_img_features,
                auxiliary_rpn_feats,
                img_metas,
                aug_gt_nbboxes,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg,
                **kwargs)
            c_rpn_losses = dict()
            for key in rpn_losses.keys():
                c_rpn_losses['c_'+str(key)] = rpn_losses[key]
            losses.update(c_rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            img_features,
            img_metas,
            proposal_list,
            gt_nbboxes,
            gt_nlabels,
            aug_img_features,
            proposal_list_c,
            aug_gt_nbboxes,
            aug_gt_nlabels,
            gt_nlabels_true,
            gt_bboxes,
            gt_labels,
            **kwargs)
        losses.update(roi_losses)
        # end of classic Faster RCNN

        auxiliary_roi_losses = self.roi_head.forward_auxiliary_train(
            auxiliary_roi_feats,
            [torch.cat(data['gt_labels']) for data in auxiliary_data_list])

        losses.update(auxiliary_roi_losses)
        # cosine sim of nbboxes and bboxes


        # shape of gt_nbboxes: list of batch length and each has: nbr bbox on image x nbr contrastive classes x [xmin ymin xmax ymax]
        # shape of gt_nlabels: list of batch length and each has: nbr bbox on image x nbr contrastive classes 
        
        # shape of nimg: list of batch length and each has: C x H x W
        # shape of img: list of batch length and each has: H x W x C

        # shape of img aug is batch x C x H x W to match img
            
        return losses
 
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )

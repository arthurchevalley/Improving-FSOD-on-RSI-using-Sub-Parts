# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_overlaps
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class CosContRoIHead_Branch(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      img_features,
                      img_metas,
                      proposal_list,
                      gt_nbboxes,
                      gt_nlabels,
                      aug_img_features,
                      c_proposal_list,
                      gt_bboxes_aug,
                      gt_labels_aug,
                      gt_labels_aug_true,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals

        # Augmented novel detections
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            aug_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    c_proposal_list[i], gt_bboxes_aug[i], gt_bboxes_ignore[i],
                    gt_labels_aug[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    c_proposal_list[i],
                    gt_bboxes_aug[i],
                    gt_labels_aug[i],
                    feats=[lvl_feat[i][None] for lvl_feat in aug_img_features])
                aug_sampling_results.append(sampling_result)

        # Novel detection
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_nbboxes[i], gt_bboxes_ignore[i],
                    gt_nlabels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_nbboxes[i],
                    gt_nlabels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                sampling_results.append(sampling_result)
        
        # Base detection , ie True ones
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            base_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                base_sampling_results.append(sampling_result)
                
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            #print("in CRFCNN ")
            bbox_results = self._bbox_forward_train(img_features, base_sampling_results, sampling_results,
                                                    gt_nbboxes, gt_nlabels,
                                                    aug_img_features, aug_sampling_results,
                                                    gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                                                    img_metas, 
                                                    gt_bboxes, gt_labels)
            losses.update(bbox_results['loss_bbox'])
            

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(img_features, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, base_rois, rois=None, x_aug=None, aug_rois=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        
        base_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], base_rois)
        if self.with_shared_head:
            base_bbox_feats = self.shared_head(base_bbox_feats)
    

        if rois is not None:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
        else:
            bbox_feats = None

        if aug_rois is not None:
            bbox_feats_aug = self.bbox_roi_extractor(
                x_aug[:self.bbox_roi_extractor.num_inputs], aug_rois)
            if self.with_shared_head:
                bbox_feats_aug = self.shared_head(bbox_feats_aug)
        else:
            bbox_feats_aug = None

        base_cls_score, base_bbox_pred, base_cont_feat, cls_score, bbox_pred, cont_feat, aug_cls_score, aug_bbox_pred, aug_cont_feat = self.bbox_head(base_bbox_feats, bbox_feats, bbox_feats_aug)

        if aug_rois is not None and rois is not None:
            base_bbox_results = dict(
                cls_score=base_cls_score, bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)
            aug_bbox_results = dict(
                cls_score=aug_cls_score, bbox_pred=aug_bbox_pred, bbox_feats=bbox_feats_aug, cont_feat=aug_cont_feat)
        
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, cont_feat=cont_feat)
            return base_bbox_results, bbox_results, aug_bbox_results
        else:
            base_bbox_results = dict(
                cls_score=base_cls_score, bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)

            return base_bbox_results


    def _bbox_forward_train(self, x, base_sampling_results, sampling_results, 
                            gt_nbboxes, gt_nlabels,
                            x_aug, aug_sampling_results, 
                            gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                            img_metas, 
                            gt_base_bbox, gt_base_labels):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in base_sampling_results])
        base_rois = base_rois.contiguous()
        
        rois = bbox2roi([res.bboxes for res in sampling_results])
        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()

        aug_rois = bbox2roi([res.bboxes for res in aug_sampling_results])
        x_aug_c = []
        for i in range(len(x_aug)):
            x_aug_c.append(x_aug[i].contiguous())
        x_aug = tuple(x_aug_c)
        aug_rois = aug_rois.contiguous()

        base_bbox_results, bbox_results, aug_bbox_results = self._bbox_forward(x, base_rois, rois, x_aug, aug_rois)

        base_bbox_targets = self.bbox_head.get_targets(base_sampling_results, gt_base_bbox, gt_base_labels, self.train_cfg)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_nbboxes, gt_nlabels, self.train_cfg)
        
        aug_bbox_targets = self.bbox_head.get_targets(aug_sampling_results, gt_bboxes_aug, gt_labels_aug, self.train_cfg)

        loss_bbox = self.bbox_head.loss(base_bbox_results['cls_score'],
                                        base_bbox_results['bbox_pred'], base_rois,
                                        *base_bbox_targets)
        
        aug_cls_score_pred = aug_bbox_results['cls_score']
        aug_bbox_score_pred = self.bbox_head.bbox_coder.decode(aug_rois[..., 1:], aug_bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])
        cls_score_pred = bbox_results['cls_score']
        bbox_score_pred = self.bbox_head.bbox_coder.decode(rois[..., 1:], bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])

        loss_bbox_cont = self.bbox_head.loss_contrast(
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_results,
                    bbox_results,
                    base_bbox_targets,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    img_metas,
                    self.bbox_head.num_classes
                )
        
        loss_bbox.update(loss_bbox_cont)
        
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels

@HEADS.register_module()
class DualCosContRoIHead_Branch(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      img_features,
                      img_metas,
                      proposal_list,
                      gt_nbboxes,
                      gt_nlabels,
                      aug_img_features,
                      c_proposal_list,
                      gt_bboxes_aug,
                      gt_labels_aug,
                      gt_labels_aug_true,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals

        # Augmented novel detections
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            aug_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    c_proposal_list[i], gt_bboxes_aug[i], gt_bboxes_ignore[i],
                    gt_labels_aug[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    c_proposal_list[i],
                    gt_bboxes_aug[i],
                    gt_labels_aug[i],
                    feats=[lvl_feat[i][None] for lvl_feat in aug_img_features])
                aug_sampling_results.append(sampling_result)

        # Novel detection
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_nbboxes[i], gt_bboxes_ignore[i],
                    gt_nlabels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_nbboxes[i],
                    gt_nlabels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                sampling_results.append(sampling_result)
        
        # Base detection , ie True ones
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            base_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                base_sampling_results.append(sampling_result)
                
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            #print("in CRFCNN ")
            bbox_results = self._bbox_forward_train(img_features, base_sampling_results, sampling_results,
                                                    gt_nbboxes, gt_nlabels,
                                                    aug_img_features, aug_sampling_results,
                                                    gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                                                    img_metas, 
                                                    gt_bboxes, gt_labels)
            losses.update(bbox_results['loss_bbox'])
            

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(img_features, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, base_rois, rois=None, x_aug=None, aug_rois=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        
        base_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], base_rois)
        if self.with_shared_head:
            base_bbox_feats = self.shared_head(base_bbox_feats)
    

        if rois is not None:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
        else:
            bbox_feats = None

        if aug_rois is not None:
            bbox_feats_aug = self.bbox_roi_extractor(
                x_aug[:self.bbox_roi_extractor.num_inputs], aug_rois)
            if self.with_shared_head:
                bbox_feats_aug = self.shared_head(bbox_feats_aug)
        else:
            bbox_feats_aug = None

        base_cls_score, base_bbox_pred, base_cont_feat, cls_score, bbox_pred, cont_feat, aug_cls_score, aug_bbox_pred, aug_cont_feat = self.bbox_head(base_bbox_feats, bbox_feats, bbox_feats_aug)

        if aug_rois is not None and rois is not None:
            base_bbox_results = dict(
                cls_score=base_cls_score, bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)
            aug_bbox_results = dict(
                cls_score=aug_cls_score, bbox_pred=aug_bbox_pred, bbox_feats=bbox_feats_aug, cont_feat=aug_cont_feat)
        
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, cont_feat=cont_feat)
            return base_bbox_results, bbox_results, aug_bbox_results
        else:
            base_bbox_results = dict(
                cls_score=base_cls_score, bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)

            return base_bbox_results

    def _bbox_forward_train(self, x, base_sampling_results, sampling_results, 
                            gt_nbboxes, gt_nlabels,
                            x_aug, aug_sampling_results, 
                            gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                            img_metas, 
                            gt_base_bbox, gt_base_labels):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in base_sampling_results])
        base_rois = base_rois.contiguous()
        proposal_ious = []
        for res in base_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        base_proposal_ious = torch.cat(proposal_ious, dim=0)


        
        rois = bbox2roi([res.bboxes for res in sampling_results])

        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        novel_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()



        aug_rois = bbox2roi([res.bboxes for res in aug_sampling_results])

        proposal_ious = []
        for res in aug_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        aug_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_aug_c = []
        for i in range(len(x_aug)):
            x_aug_c.append(x_aug[i].contiguous())
        x_aug = tuple(x_aug_c)
        aug_rois = aug_rois.contiguous()






        base_bbox_results, bbox_results, aug_bbox_results = self._bbox_forward(x, base_rois, rois, x_aug, aug_rois)

        base_bbox_targets = self.bbox_head.get_targets(base_sampling_results, gt_base_bbox, gt_base_labels, self.train_cfg)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_nbboxes, gt_nlabels, self.train_cfg)
        
        aug_bbox_targets = self.bbox_head.get_targets(aug_sampling_results, gt_bboxes_aug, gt_labels_aug, self.train_cfg)

        

        loss_bbox = self.bbox_head.loss(base_bbox_results['cls_score'],
                                        base_bbox_results['bbox_pred'], base_rois,
                                        *base_bbox_targets)
        
        aug_cls_score_pred = aug_bbox_results['cls_score']
        aug_bbox_score_pred = self.bbox_head.bbox_coder.decode(aug_rois[..., 1:], aug_bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])
        cls_score_pred = bbox_results['cls_score']
        bbox_score_pred = self.bbox_head.bbox_coder.decode(rois[..., 1:], bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])

        loss_bbox_cont = self.bbox_head.loss_contrast(
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_results,
                    bbox_results,
                    base_bbox_targets,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    img_metas,
                    self.bbox_head.num_classes,
                    base_proposal_ious,
                    novel_proposal_ious,
                    aug_proposal_ious

                )

        
        loss_bbox.update(loss_bbox_cont)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

@HEADS.register_module()
class DualCosContRoIHead_Branch_novcls(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      img_features,
                      img_metas,
                      proposal_list,
                      gt_nbboxes,
                      gt_nlabels,
                      aug_img_features,
                      c_proposal_list,
                      gt_bboxes_aug,
                      gt_labels_aug,
                      gt_labels_aug_true,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals

        # Augmented novel detections
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            aug_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    c_proposal_list[i], gt_bboxes_aug[i], gt_bboxes_ignore[i],
                    gt_labels_aug[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    c_proposal_list[i],
                    gt_bboxes_aug[i],
                    gt_labels_aug[i],
                    feats=[lvl_feat[i][None] for lvl_feat in aug_img_features])
                aug_sampling_results.append(sampling_result)

        # Novel detection
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_nbboxes[i], gt_bboxes_ignore[i],
                    gt_nlabels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_nbboxes[i],
                    gt_nlabels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                sampling_results.append(sampling_result)
        
        # Base detection , ie True ones
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            base_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                base_sampling_results.append(sampling_result)
                
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            #print("in CRFCNN ")
            bbox_results = self._bbox_forward_train(img_features, base_sampling_results, sampling_results,
                                                    gt_nbboxes, gt_nlabels,
                                                    aug_img_features, aug_sampling_results,
                                                    gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                                                    img_metas, 
                                                    gt_bboxes, gt_labels)
            losses.update(bbox_results['loss_bbox'])
            

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(img_features, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, base_rois, rois=None, x_aug=None, aug_rois=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        
        base_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], base_rois)
        if self.with_shared_head:
            base_bbox_feats = self.shared_head(base_bbox_feats)
    
        
        if rois is not None:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
        else:
            bbox_feats = None

        if aug_rois is not None:
            bbox_feats_aug = self.bbox_roi_extractor(
                x_aug[:self.bbox_roi_extractor.num_inputs], aug_rois)
            if self.with_shared_head:
                bbox_feats_aug = self.shared_head(bbox_feats_aug)
        else:
            bbox_feats_aug = None

        #print(f'roi feat base, novel, aug {base_bbox_feats.shape, bbox_feats.shape, bbox_feats_aug.shape}')

        base_cls_score, base_bbox_pred, base_cont_feat, cls_score, bbox_pred, cont_feat, aug_cls_score, aug_bbox_pred, aug_cont_feat = self.bbox_head(base_bbox_feats, bbox_feats, bbox_feats_aug)

        if aug_rois is not None and rois is not None:
            base_bbox_results = dict(
                cls_score=base_cls_score, bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)
            aug_bbox_results = dict(
                cls_score=aug_cls_score, bbox_pred=aug_bbox_pred, bbox_feats=bbox_feats_aug, cont_feat=aug_cont_feat)
        
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, cont_feat=cont_feat)
            return base_bbox_results, bbox_results, aug_bbox_results
        else:
            base_bbox_results = dict(
                cls_score=base_cls_score, bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)

            return base_bbox_results

    def _bbox_forward_train(self, x, base_sampling_results, sampling_results, 
                            gt_nbboxes, gt_nlabels,
                            x_aug, aug_sampling_results, 
                            gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                            img_metas, 
                            gt_base_bbox, gt_base_labels):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in base_sampling_results])
        base_rois = base_rois.contiguous()
        proposal_ious = []
        for res in base_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        base_proposal_ious = torch.cat(proposal_ious, dim=0)


        rois = bbox2roi([res.bboxes for res in sampling_results])

        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        novel_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()



        aug_rois = bbox2roi([res.bboxes for res in aug_sampling_results])

        proposal_ious = []
        for res in aug_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        aug_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_aug_c = []
        for i in range(len(x_aug)):
            x_aug_c.append(x_aug[i].contiguous())
        x_aug = tuple(x_aug_c)
        aug_rois = aug_rois.contiguous()


        base_bbox_results, bbox_results, aug_bbox_results = self._bbox_forward(x, base_rois, rois, x_aug, aug_rois)
        base_bbox_targets = self.bbox_head.get_targets(base_sampling_results, gt_base_bbox, gt_base_labels, self.train_cfg)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_nbboxes, gt_nlabels, self.train_cfg)
        
        aug_bbox_targets = self.bbox_head.get_targets(aug_sampling_results, gt_bboxes_aug, gt_labels_aug, self.train_cfg)

        gt_nlabels_tmp = []
        gt_labels_aug_tmp = []

        for i in range(len(gt_labels_aug_true)):
            gt_labels_aug_tmp += gt_labels_aug_true[i].tolist()
            gt_nlabels_tmp += gt_nlabels[i].tolist()
        
        classes_eq = {gt_nlabels_tmp[i]: gt_labels_aug_tmp[i] for i in range(len(gt_nlabels_tmp))}

        #loss_bbox = self.bbox_head.loss(base_bbox_results['cls_score'],
         #                               base_bbox_results['bbox_pred'], 
          #                              base_rois,
           #                             *base_bbox_targets)
        #print(f'loss {loss_bbox}')
        if classes_eq is not None:
            classes_eq[self.bbox_head.num_classes]=self.bbox_head.num_classes
            if (bbox_targets[0] > self.bbox_head.num_classes).any():
                labels_novel = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cls_score'].device)
            if (aug_bbox_targets[0] > self.bbox_head.num_classes).any():
                aug_labels_novel = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0]]).to(aug_bbox_results['cls_score'].device)
        
            redo_bbox_targets = (labels_novel, 0.5*bbox_targets[1].clone(), bbox_targets[2].clone(), .5*bbox_targets[3].clone())
            redo_aug_bbox_targets = (aug_labels_novel, 0.5*aug_bbox_targets[1].clone(), aug_bbox_targets[2].clone(), .5*aug_bbox_targets[3].clone())
        #print(bbox_targets[1].shape)
        #fuse_bbox_targets = (torch.hstack((base_bbox_targets[0], labels_novel, aug_labels_novel)),
        #                    torch.hstack((base_bbox_targets[1], 0.5*bbox_targets[1], 0.5*aug_bbox_targets[1])),
        #                    torch.vstack((base_bbox_targets[2], bbox_targets[2], .5*bbox_targets[3])),
         #                   torch.vstack((base_bbox_targets[3], aug_bbox_targets[2], .5*aug_bbox_targets[3]))
         #                   )
        #fuse_cls_score = torch.vstack((base_bbox_results['cls_score'], bbox_results['cls_score'], aug_bbox_results['cls_score']))
        #fuse_bbox_pred = torch.vstack((base_bbox_results['bbox_pred'], bbox_results['bbox_pred'], aug_bbox_results['bbox_pred']))
        #fuse_iou = torch.vstack((base_rois, rois, aug_rois))
        #print(f'lbl {fuse_bbox_targets[0].shape, fuse_cls_score.shape}')
        #print(f'bbox {fuse_bbox_targets[2].shape, fuse_bbox_pred.shape}')
        
        #loss_bbox = self.bbox_head.loss(fuse_cls_score,
        #                                fuse_bbox_pred, 
        #                                fuse_iou,
        #                                *fuse_bbox_targets)
        #del fuse_bbox_targets
        ##del fuse_cls_score
        #del fuse_bbox_pred
        #del fuse_iou
        
        loss_bbox_base = self.bbox_head.loss(base_bbox_results['cls_score'],
                                        base_bbox_results['bbox_pred'], 
                                        base_rois,
                                        *base_bbox_targets)
        loss_bbox_novel = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], 
                                        rois,
                                        *redo_bbox_targets)
                                        
        loss_bbox_aug = self.bbox_head.loss(aug_bbox_results['cls_score'],
                                        aug_bbox_results['bbox_pred'], 
                                        aug_rois,
                                        *redo_aug_bbox_targets)
                                        
        loss_fuse = sum([loss_bbox_base['loss_cls'],loss_bbox_novel['loss_cls'],loss_bbox_aug['loss_cls']])
        loss_bbox_base.update(loss_cls=loss_fuse)

        loss_fuse = sum([loss_bbox_base['loss_bbox'],loss_bbox_novel['loss_bbox'],loss_bbox_aug['loss_bbox']])
        loss_bbox_base.update(loss_bbox=loss_fuse)

        loss_fuse = sum([loss_bbox_base['acc'],loss_bbox_novel['acc'],loss_bbox_aug['acc']])/3
        loss_bbox_base.update(acc=loss_fuse)


        aug_cls_score_pred = aug_bbox_results['cls_score']
        aug_bbox_score_pred = self.bbox_head.bbox_coder.decode(aug_rois[..., 1:].clone(), aug_bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])
        cls_score_pred = bbox_results['cls_score']
        bbox_score_pred = self.bbox_head.bbox_coder.decode(rois[..., 1:].clone(), bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])

        loss_bbox_cont = self.bbox_head.loss_contrast(
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_results,
                    bbox_results,
                    base_bbox_targets,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    img_metas,
                    self.bbox_head.num_classes,
                    base_proposal_ious,
                    novel_proposal_ious,
                    aug_proposal_ious

                )

        
        loss_bbox_base.update(loss_bbox_cont)

        bbox_results.update(loss_bbox=loss_bbox_base)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

@HEADS.register_module()
class DualCosContRoIHead_Branch_novcls_separate(DualCosContRoIHead_Branch_novcls):
    """Simplest base roi head including one bbox head and one mask head."""

    def _bbox_forward_train(self, x, base_sampling_results, sampling_results, 
                            gt_nbboxes, gt_nlabels,
                            x_aug, aug_sampling_results, 
                            gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                            img_metas, 
                            gt_base_bbox, gt_base_labels):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in base_sampling_results])
        base_rois = base_rois.contiguous()
        proposal_ious = []
        for res in base_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        base_proposal_ious = torch.cat(proposal_ious, dim=0)


        rois = bbox2roi([res.bboxes for res in sampling_results])

        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        novel_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()



        aug_rois = bbox2roi([res.bboxes for res in aug_sampling_results])

        proposal_ious = []
        for res in aug_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        aug_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_aug_c = []
        for i in range(len(x_aug)):
            x_aug_c.append(x_aug[i].contiguous())
        x_aug = tuple(x_aug_c)
        aug_rois = aug_rois.contiguous()


        base_bbox_results, bbox_results, aug_bbox_results = self._bbox_forward(x, base_rois, rois, x_aug, aug_rois)
        base_bbox_targets = self.bbox_head.get_targets(base_sampling_results, gt_base_bbox, gt_base_labels, self.train_cfg)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_nbboxes, gt_nlabels, self.train_cfg)
        
        aug_bbox_targets = self.bbox_head.get_targets(aug_sampling_results, gt_bboxes_aug, gt_labels_aug, self.train_cfg)

        gt_nlabels_tmp = []
        gt_labels_aug_tmp = []

        for i in range(len(gt_labels_aug_true)):
            gt_labels_aug_tmp += gt_labels_aug_true[i].tolist()
            gt_nlabels_tmp += gt_nlabels[i].tolist()
        
        classes_eq = {gt_nlabels_tmp[i]: gt_labels_aug_tmp[i] for i in range(len(gt_nlabels_tmp))}

        
        if classes_eq is not None:
            classes_eq[self.bbox_head.num_classes]=self.bbox_head.num_classes
            if (bbox_targets[0] > self.bbox_head.num_classes).any():
                labels_novel = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cls_score'].device)
            if (aug_bbox_targets[0] > self.bbox_head.num_classes).any():
                aug_labels_novel = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0]]).to(aug_bbox_results['cls_score'].device)
        
            redo_bbox_targets = (labels_novel, 0.5*bbox_targets[1].clone(), bbox_targets[2].clone(), .5*bbox_targets[3].clone())
            redo_aug_bbox_targets = (aug_labels_novel, 0.5*aug_bbox_targets[1].clone(), aug_bbox_targets[2].clone(), .5*aug_bbox_targets[3].clone())
        
        loss_roi = dict()
        loss_bbox_base = self.bbox_head.loss(base_bbox_results['cls_score'],
                                        base_bbox_results['bbox_pred'], 
                                        base_rois,
                                        *base_bbox_targets)
        loss_roi.update(loss_bbox_base)
        loss_bbox_novel = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], 
                                        rois,
                                        *redo_bbox_targets)
        tmp_loss = {}
        for key in loss_bbox_novel.keys():
            tmp_loss['subregion_'+key] = loss_bbox_novel[key]
            
        loss_roi.update(tmp_loss)
        loss_bbox_aug = self.bbox_head.loss(aug_bbox_results['cls_score'],
                                        aug_bbox_results['bbox_pred'], 
                                        aug_rois,
                                        *redo_aug_bbox_targets)
        tmp_loss = {}
        for key in loss_bbox_aug.keys():
            tmp_loss['augmented_'+key] = loss_bbox_aug[key]
            
        loss_roi.update(tmp_loss)
        del tmp_loss
        #loss_fuse = sum([loss_bbox_base['loss_cls'],loss_bbox_novel['loss_cls'],loss_bbox_aug['loss_cls']])
        #loss_bbox_base.update(loss_cls=loss_fuse)

        #loss_fuse = sum([loss_bbox_base['loss_bbox'],loss_bbox_novel['loss_bbox'],loss_bbox_aug['loss_bbox']])
        #loss_bbox_base.update(loss_bbox=loss_fuse)

        #loss_fuse = sum([loss_bbox_base['acc'],loss_bbox_novel['acc'],loss_bbox_aug['acc']])/3
        #loss_bbox_base.update(acc=loss_fuse)


        aug_cls_score_pred = aug_bbox_results['cls_score']
        aug_bbox_score_pred = self.bbox_head.bbox_coder.decode(aug_rois[..., 1:].clone(), aug_bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])
        cls_score_pred = bbox_results['cls_score']
        bbox_score_pred = self.bbox_head.bbox_coder.decode(rois[..., 1:].clone(), bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])

        loss_bbox_cont = self.bbox_head.loss_contrast(
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_results,
                    bbox_results,
                    base_bbox_targets,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    img_metas,
                    self.bbox_head.num_classes,
                    base_proposal_ious,
                    novel_proposal_ious,
                    aug_proposal_ious

                )

        
        loss_roi.update(loss_bbox_cont)

        bbox_results.update(loss_bbox=loss_roi)
        return bbox_results

@HEADS.register_module()
class DualCosContRoIHead_Branch_novcls_nobbox_separate(DualCosContRoIHead_Branch_novcls):
    """Simplest base roi head including one bbox head and one mask head."""

    def _bbox_forward_train(self, x, base_sampling_results, sampling_results, 
                            gt_nbboxes, gt_nlabels,
                            x_aug, aug_sampling_results, 
                            gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                            img_metas, 
                            gt_base_bbox, gt_base_labels):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in base_sampling_results])
        base_rois = base_rois.contiguous()
        proposal_ious = []
        for res in base_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        base_proposal_ious = torch.cat(proposal_ious, dim=0)

        rois = bbox2roi([res.bboxes for res in sampling_results])

        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        novel_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()



        aug_rois = bbox2roi([res.bboxes for res in aug_sampling_results])

        proposal_ious = []
        for res in aug_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        aug_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_aug_c = []
        for i in range(len(x_aug)):
            x_aug_c.append(x_aug[i].contiguous())
        x_aug = tuple(x_aug_c)
        aug_rois = aug_rois.contiguous()

        #print(f'roi shape base, novel aug {base_rois.shape, aug_rois.shape, rois.shape}')

        base_bbox_results, bbox_results, aug_bbox_results = self._bbox_forward(x, base_rois, rois, x_aug, aug_rois)
        base_bbox_targets = self.bbox_head.get_targets(base_sampling_results, gt_base_bbox, gt_base_labels, self.train_cfg)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_nbboxes, gt_nlabels, self.train_cfg)
        
        aug_bbox_targets = self.bbox_head.get_targets(aug_sampling_results, gt_bboxes_aug, gt_labels_aug, self.train_cfg)

        gt_nlabels_tmp = []
        gt_labels_aug_tmp = []

        for i in range(len(gt_labels_aug_true)):
            gt_labels_aug_tmp += gt_labels_aug_true[i].tolist()
            gt_nlabels_tmp += gt_nlabels[i].tolist()
        
        classes_eq = {gt_nlabels_tmp[i]: gt_labels_aug_tmp[i] for i in range(len(gt_nlabels_tmp))}

        
        if classes_eq is not None:
            classes_eq[self.bbox_head.num_classes]=self.bbox_head.num_classes
            if (bbox_targets[0] > self.bbox_head.num_classes).any():
                labels_novel = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cls_score'].device)
            if (aug_bbox_targets[0] > self.bbox_head.num_classes).any():
                aug_labels_novel = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0]]).to(aug_bbox_results['cls_score'].device)
        
            redo_bbox_targets = (labels_novel, 0.5*bbox_targets[1].clone(), bbox_targets[2].clone(), .5*bbox_targets[3].clone())
            redo_aug_bbox_targets = (aug_labels_novel, 0.5*aug_bbox_targets[1].clone(), aug_bbox_targets[2].clone(), .5*aug_bbox_targets[3].clone())

        loss_roi = dict()
        
        loss_bbox_base = self.bbox_head.loss(base_bbox_results['cls_score'],
                                        base_bbox_results['bbox_pred'], 
                                        base_rois,
                                        *base_bbox_targets)

        loss_roi.update(loss_bbox_base)
        loss_bbox_novel = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], 
                                        rois,
                                        *redo_bbox_targets)
        tmp_loss = {}
        for key in loss_bbox_novel.keys():
            if key == 'loss_bbox':
                continue
            tmp_loss['subregion_'+key] = loss_bbox_novel[key]
            
        loss_roi.update(tmp_loss)
        loss_bbox_aug = self.bbox_head.loss(aug_bbox_results['cls_score'],
                                        aug_bbox_results['bbox_pred'], 
                                        aug_rois,
                                        *redo_aug_bbox_targets)
        tmp_loss = {}
        for key in loss_bbox_aug.keys():
            if key == 'loss_bbox':
                continue
            tmp_loss['augmented_'+key] = loss_bbox_aug[key]
            
        loss_roi.update(tmp_loss)
        del tmp_loss
        #loss_fuse = sum([loss_bbox_base['loss_cls'],loss_bbox_novel['loss_cls'],loss_bbox_aug['loss_cls']])
        #loss_bbox_base.update(loss_cls=loss_fuse)

        #loss_fuse = sum([loss_bbox_base['loss_bbox'],loss_bbox_novel['loss_bbox'],loss_bbox_aug['loss_bbox']])
        #loss_bbox_base.update(loss_bbox=loss_fuse)

        #loss_fuse = sum([loss_bbox_base['acc'],loss_bbox_novel['acc'],loss_bbox_aug['acc']])/3
        #loss_bbox_base.update(acc=loss_fuse)


        aug_cls_score_pred = aug_bbox_results['cls_score']
        aug_bbox_score_pred = self.bbox_head.bbox_coder.decode(aug_rois[..., 1:].clone(), aug_bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])
        cls_score_pred = bbox_results['cls_score']
        bbox_score_pred = self.bbox_head.bbox_coder.decode(rois[..., 1:].clone(), bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])

        loss_bbox_cont = self.bbox_head.loss_contrast(
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_results,
                    bbox_results,
                    base_bbox_targets,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    img_metas,
                    self.bbox_head.num_classes,
                    base_proposal_ious,
                    novel_proposal_ious,
                    aug_proposal_ious
                )
        
        loss_roi.update(loss_bbox_cont)
       

        bbox_results.update(loss_bbox=loss_roi)
        return bbox_results

@HEADS.register_module()
class DualCosContRoIHead_Branch_novcls_nobbox(DualCosContRoIHead_Branch_novcls):
    """Simplest base roi head including one bbox head and one mask head."""

    def _bbox_forward_train(self, x, base_sampling_results, sampling_results, 
                            gt_nbboxes, gt_nlabels,
                            x_aug, aug_sampling_results, 
                            gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                            img_metas, 
                            gt_base_bbox, gt_base_labels):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in base_sampling_results])
        base_rois = base_rois.contiguous()
        proposal_ious = []
        for res in base_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        base_proposal_ious = torch.cat(proposal_ious, dim=0)


        rois = bbox2roi([res.bboxes for res in sampling_results])

        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        novel_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()



        aug_rois = bbox2roi([res.bboxes for res in aug_sampling_results])

        proposal_ious = []
        for res in aug_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        aug_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_aug_c = []
        for i in range(len(x_aug)):
            x_aug_c.append(x_aug[i].contiguous())
        x_aug = tuple(x_aug_c)
        aug_rois = aug_rois.contiguous()


        base_bbox_results, bbox_results, aug_bbox_results = self._bbox_forward(x, base_rois, rois, x_aug, aug_rois)
        base_bbox_targets = self.bbox_head.get_targets(base_sampling_results, gt_base_bbox, gt_base_labels, self.train_cfg)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_nbboxes, gt_nlabels, self.train_cfg)
        
        aug_bbox_targets = self.bbox_head.get_targets(aug_sampling_results, gt_bboxes_aug, gt_labels_aug, self.train_cfg)

        gt_nlabels_tmp = []
        gt_labels_aug_tmp = []

        for i in range(len(gt_labels_aug_true)):
            gt_labels_aug_tmp += gt_labels_aug_true[i].tolist()
            gt_nlabels_tmp += gt_nlabels[i].tolist()
        
        classes_eq = {gt_nlabels_tmp[i]: gt_labels_aug_tmp[i] for i in range(len(gt_nlabels_tmp))}

        #loss_bbox = self.bbox_head.loss(base_bbox_results['cls_score'],
         #                               base_bbox_results['bbox_pred'], 
          #                              base_rois,
           #                             *base_bbox_targets)
        #print(f'loss {loss_bbox}')
        if classes_eq is not None:
            classes_eq[self.bbox_head.num_classes]=self.bbox_head.num_classes
            if (bbox_targets[0] > self.bbox_head.num_classes).any():
                labels_novel = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['cls_score'].device)
            if (aug_bbox_targets[0] > self.bbox_head.num_classes).any():
                aug_labels_novel = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0]]).to(aug_bbox_results['cls_score'].device)
        
            redo_bbox_targets = (labels_novel, 0.5*bbox_targets[1].clone(), bbox_targets[2].clone(), .5*bbox_targets[3].clone())
            redo_aug_bbox_targets = (aug_labels_novel, 0.5*aug_bbox_targets[1].clone(), aug_bbox_targets[2].clone(), .5*aug_bbox_targets[3].clone())
        #print(bbox_targets[1].shape)
        #fuse_bbox_targets = (torch.hstack((base_bbox_targets[0], labels_novel, aug_labels_novel)),
        #                    torch.hstack((base_bbox_targets[1], 0.5*bbox_targets[1], 0.5*aug_bbox_targets[1])),
        #                    torch.vstack((base_bbox_targets[2], bbox_targets[2], .5*bbox_targets[3])),
         #                   torch.vstack((base_bbox_targets[3], aug_bbox_targets[2], .5*aug_bbox_targets[3]))
         #                   )
        #fuse_cls_score = torch.vstack((base_bbox_results['cls_score'], bbox_results['cls_score'], aug_bbox_results['cls_score']))
        #fuse_bbox_pred = torch.vstack((base_bbox_results['bbox_pred'], bbox_results['bbox_pred'], aug_bbox_results['bbox_pred']))
        #fuse_iou = torch.vstack((base_rois, rois, aug_rois))
        #print(f'lbl {fuse_bbox_targets[0].shape, fuse_cls_score.shape}')
        #print(f'bbox {fuse_bbox_targets[2].shape, fuse_bbox_pred.shape}')
        
        #loss_bbox = self.bbox_head.loss(fuse_cls_score,
        #                                fuse_bbox_pred, 
        #                                fuse_iou,
        #                                *fuse_bbox_targets)
        #del fuse_bbox_targets
        ##del fuse_cls_score
        #del fuse_bbox_pred
        #del fuse_iou
        
        loss_bbox_base = self.bbox_head.loss(base_bbox_results['cls_score'],
                                        base_bbox_results['bbox_pred'], 
                                        base_rois,
                                        *base_bbox_targets)
                                        
        loss_bbox_novel = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], 
                                        rois,
                                        *redo_bbox_targets)
                                        
        loss_bbox_aug = self.bbox_head.loss(aug_bbox_results['cls_score'],
                                        aug_bbox_results['bbox_pred'], 
                                        aug_rois,
                                        *redo_aug_bbox_targets)
                                        
        loss_fuse = sum([loss_bbox_base['loss_cls'],loss_bbox_novel['loss_cls'],loss_bbox_aug['loss_cls']])
        loss_bbox_base.update(loss_cls=loss_fuse)


        aug_cls_score_pred = aug_bbox_results['cls_score']
        aug_bbox_score_pred = self.bbox_head.bbox_coder.decode(aug_rois[..., 1:].clone(), aug_bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])
        cls_score_pred = bbox_results['cls_score']
        bbox_score_pred = self.bbox_head.bbox_coder.decode(rois[..., 1:].clone(), bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])

        loss_bbox_cont = self.bbox_head.loss_contrast(
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_results,
                    bbox_results,
                    base_bbox_targets,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    img_metas,
                    self.bbox_head.num_classes,
                    base_proposal_ious,
                    novel_proposal_ious,
                    aug_proposal_ious

                )

        
        loss_bbox_base.update(loss_bbox_cont)

        bbox_results.update(loss_bbox=loss_bbox_base)
        return bbox_results

@HEADS.register_module()
class Agnostic_DualCosContRoIHead_Branch_novcls_nobbox_separate(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      img_features,
                      img_metas,
                      proposal_list,
                      gt_nbboxes,
                      gt_nlabels,
                      aug_img_features,
                      c_proposal_list,
                      gt_bboxes_aug,
                      gt_labels_aug,
                      gt_labels_aug_true,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals

        # Augmented novel detections
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            aug_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    c_proposal_list[i], gt_bboxes_aug[i], gt_bboxes_ignore[i],
                    gt_labels_aug[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    c_proposal_list[i],
                    gt_bboxes_aug[i],
                    gt_labels_aug[i],
                    feats=[lvl_feat[i][None] for lvl_feat in aug_img_features])
                aug_sampling_results.append(sampling_result)

        # Novel detection
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_nbboxes[i], gt_bboxes_ignore[i],
                    gt_nlabels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_nbboxes[i],
                    gt_nlabels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                sampling_results.append(sampling_result)
        
        # Base detection , ie True ones
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            base_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                base_sampling_results.append(sampling_result)
                
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            #print("in CRFCNN ")
            bbox_results = self._bbox_forward_train(img_features, base_sampling_results, sampling_results,
                                                    gt_nbboxes, gt_nlabels,
                                                    aug_img_features, aug_sampling_results,
                                                    gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                                                    img_metas, 
                                                    gt_bboxes, gt_labels)
            losses.update(bbox_results['loss_bbox'])
            

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(img_features, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, base_rois, rois=None, x_aug=None, aug_rois=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        
        base_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], base_rois)
        if self.with_shared_head:
            base_bbox_feats = self.shared_head(base_bbox_feats)
    
        
        if rois is not None:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
        else:
            bbox_feats = None

        if aug_rois is not None:
            bbox_feats_aug = self.bbox_roi_extractor(
                x_aug[:self.bbox_roi_extractor.num_inputs], aug_rois)
            if self.with_shared_head:
                bbox_feats_aug = self.shared_head(bbox_feats_aug)
        else:
            bbox_feats_aug = None

        #print(f'roi feat base, novel, aug {base_bbox_feats.shape, bbox_feats.shape, bbox_feats_aug.shape}')

        base_bbox_pred, base_cont_feat, bbox_pred, cont_feat, aug_bbox_pred, aug_cont_feat = self.bbox_head(base_bbox_feats, bbox_feats, bbox_feats_aug)

        if aug_rois is not None and rois is not None:
            base_bbox_results = dict(
                bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)
            aug_bbox_results = dict(
                bbox_pred=aug_bbox_pred, bbox_feats=bbox_feats_aug, cont_feat=aug_cont_feat)
        
            bbox_results = dict(
                bbox_pred=bbox_pred, bbox_feats=bbox_feats, cont_feat=cont_feat)
            return base_bbox_results, bbox_results, aug_bbox_results
        else:
            base_bbox_results = dict(
                bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats, cont_feat=base_cont_feat)

            return base_bbox_results

    def _bbox_forward_train(self, x, base_sampling_results, sampling_results, 
                            gt_nbboxes, gt_nlabels,
                            x_aug, aug_sampling_results, 
                            gt_bboxes_aug, gt_labels_aug, gt_labels_aug_true,
                            img_metas, 
                            gt_base_bbox, gt_base_labels):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in base_sampling_results])
        base_rois = base_rois.contiguous()
        proposal_ious = []
        for res in base_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        base_proposal_ious = torch.cat(proposal_ious, dim=0)


        rois = bbox2roi([res.bboxes for res in sampling_results])

        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        novel_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()



        aug_rois = bbox2roi([res.bboxes for res in aug_sampling_results])

        proposal_ious = []
        for res in aug_sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        aug_proposal_ious = torch.cat(proposal_ious, dim=0)

        x_aug_c = []
        for i in range(len(x_aug)):
            x_aug_c.append(x_aug[i].contiguous())
        x_aug = tuple(x_aug_c)
        aug_rois = aug_rois.contiguous()

        #print(f'roi shape base, novel aug {base_rois.shape, aug_rois.shape, rois.shape}')

        base_bbox_results, bbox_results, aug_bbox_results = self._bbox_forward(x, base_rois, rois, x_aug, aug_rois)
        base_bbox_targets = self.bbox_head.get_targets(base_sampling_results, gt_base_bbox, gt_base_labels, self.train_cfg)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_nbboxes, gt_nlabels, self.train_cfg)
        
        aug_bbox_targets = self.bbox_head.get_targets(aug_sampling_results, gt_bboxes_aug, gt_labels_aug, self.train_cfg)

        gt_nlabels_tmp = []
        gt_labels_aug_tmp = []

        for i in range(len(gt_labels_aug_true)):
            gt_labels_aug_tmp += gt_labels_aug_true[i].tolist()
            gt_nlabels_tmp += gt_nlabels[i].tolist()
        
        classes_eq = {gt_nlabels_tmp[i]: gt_labels_aug_tmp[i] for i in range(len(gt_nlabels_tmp))}

        
        if classes_eq is not None:
            classes_eq[self.bbox_head.num_classes]=self.bbox_head.num_classes
            if (bbox_targets[0] > self.bbox_head.num_classes).any():
                labels_novel = torch.tensor([classes_eq[int(i)] for i in bbox_targets[0]]).to(bbox_results['bbox_pred'].device)
            if (aug_bbox_targets[0] > self.bbox_head.num_classes).any():
                aug_labels_novel = torch.tensor([classes_eq[int(i)] for i in aug_bbox_targets[0]]).to(aug_bbox_results['bbox_pred'].device)
        
            redo_bbox_targets = (labels_novel, 0.5*bbox_targets[1].clone(), bbox_targets[2].clone(), .5*bbox_targets[3].clone())
            redo_aug_bbox_targets = (aug_labels_novel, 0.5*aug_bbox_targets[1].clone(), aug_bbox_targets[2].clone(), .5*aug_bbox_targets[3].clone())

        loss_roi = dict()
        loss_bbox_base = self.bbox_head.loss(None,
                                        base_bbox_results['bbox_pred'], 
                                        base_rois,
                                        *base_bbox_targets)
        loss_roi.update(loss_bbox_base)
        aug_bbox_score_pred = self.bbox_head.bbox_coder.decode(aug_rois[..., 1:].clone(), aug_bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])
        bbox_score_pred = self.bbox_head.bbox_coder.decode(rois[..., 1:].clone(), bbox_results['bbox_pred'], max_shape=img_metas[0]['img_shape'])

        loss_bbox_cont = self.bbox_head.loss_contrast(
                    gt_labels_aug,
                    gt_labels_aug_true,
                    gt_nlabels,
                    base_bbox_results,
                    bbox_results,
                    base_bbox_targets,
                    bbox_targets,
                    aug_bbox_results,
                    aug_bbox_targets,
                    bbox_score_pred,
                    aug_bbox_score_pred,
                    img_metas,
                    self.bbox_head.num_classes,
                    base_proposal_ious,
                    novel_proposal_ious,
                    aug_proposal_ious
                )
        
        loss_roi.update(loss_bbox_cont)
       

        bbox_results.update(loss_bbox=loss_roi)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results
    
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

@HEADS.register_module()
class Agnostic_RoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      img_features,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals
        
        # Base detection , ie True ones
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            base_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_features])
                base_sampling_results.append(sampling_result)
                
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            #print("in CRFCNN ")
            bbox_results = self._bbox_forward_train(img_features, base_sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(img_features, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        
        base_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            base_bbox_feats = self.shared_head(bbox_feats)
    

        #print(f'roi feat base, novel, aug {base_bbox_feats.shape, bbox_feats.shape, bbox_feats_aug.shape}')

        base_bbox_pred = self.bbox_head(base_bbox_feats)

        bbox_results = dict(bbox_pred=base_bbox_pred, bbox_feats=base_bbox_feats)

        return bbox_results
            
    def _bbox_forward_train(self, x, sampling_results, 
                            gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""

        base_rois = bbox2roi([res.bboxes for res in sampling_results])
        base_rois = base_rois.contiguous()
        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        base_proposal_ious = torch.cat(proposal_ious, dim=0)


        rois = bbox2roi([res.bboxes for res in sampling_results])

        x_c = []
        for i in range(len(x)):
            x_c.append(x[i].contiguous())
        x = tuple(x_c)
        rois = rois.contiguous()

        #print(f'roi shape base, novel aug {base_rois.shape, aug_rois.shape, rois.shape}')

        bbox_results = self._bbox_forward(x, rois)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        
        loss_roi = dict()
        loss_bbox_base = self.bbox_head.loss(None,
                                            bbox_results['bbox_pred'], 
                                            base_rois,
                                            *bbox_targets)
                   
        loss_roi.update(loss_bbox_base)
       

        bbox_results.update(loss_bbox=loss_roi)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results
    

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))


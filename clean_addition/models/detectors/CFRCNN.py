# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models import BaseDetector


import mmcv
import numpy as np
from mmcv.runner import BaseModule, auto_fp16

@DETECTORS.register_module()
class CFRCNN(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 c_rpn_head = None,
                 c_roi_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CFRCNN, self).__init__(
            init_cfg=init_cfg)
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

        if c_rpn_head is not None:
            c_rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            c_rpn_head_ = c_rpn_head.copy()
            c_rpn_head_.update(train_cfg=c_rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.c_rpn_head = build_head(c_rpn_head_)

        if c_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            c_rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            c_roi_head.update(train_cfg=c_rcnn_train_cfg)
            c_roi_head.update(test_cfg=test_cfg.rcnn)
            c_roi_head.pretrained = pretrained
            self.c_roi_head = build_head(c_roi_head)
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
    
    @property
    def with_c_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.c_rpn_head is not None

    @property
    def with_c_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.c_roi_head is not None

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
    
    def _nBBOX_augment(self, img, bboxes, labels):

        import albumentations as A

        transform = A.ReplayCompose(
                    [
                        A.HorizontalFlip(p=0.8),
                        A.VerticalFlip(p=0.8),
                        #A.Affine (scale=(0.9, 1.1), rotate=(0), p=0.8),
                       #A.RandomBrightnessContrast(p=0.3),
                    ],
                        bbox_params=
                            A.BboxParams(format='pascal_voc', min_visibility=0.5, label_fields=['class_labels']),
                    )
                    
        list_of_bbox = bboxes
        list_of_labels = labels
        
        aug_img = []
        aug_bboxes = []
        aug_labels = []
        # for each image
        for batch_id in range(len(list_of_bbox)):
            aug_bboxes_tmp = []
            aug_labels_tmp = []

            # for each object on the image
            for bbox_id in range(list_of_labels[batch_id].shape[0]):
                current_labels = list_of_labels[batch_id][bbox_id]
                
                current_bboxes = list_of_bbox[batch_id][bbox_id]
                
                current_img = img[batch_id].cpu().detach().numpy()
                current_labels = current_labels.cpu().detach().numpy()
                current_bboxes = current_bboxes.cpu().detach().numpy()
                
                if not bbox_id:
                    transformed = transform(image=current_img, bboxes=current_bboxes, class_labels=current_labels)
                    applied_transform = []
                    to_print = transformed['replay']['transforms']

                    for tran in transformed['replay']['transforms']:
                        if tran['applied']:
                            applied_transform.append([tran['__class_fullname__'], tran['params'], img[batch_id].shape])
                    
                else:
                    transformed = A.ReplayCompose.replay(transformed['replay'], image=current_img, bboxes=current_bboxes, class_labels=current_labels)
                    to_print = transformed['replay']['transforms']

                aug_bboxes_tmp.append(transformed['bboxes'])
                aug_labels_tmp.append(transformed['class_labels'])

            aug_bboxes.append(aug_bboxes_tmp)
            aug_labels.append(aug_labels_tmp)
            aug_img.append(torch.tensor(transformed['image']))

        return aug_img, aug_bboxes, aug_labels, applied_transform

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_nbboxes, 
                      gt_nlabels,
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
        #device = torch.device('cuda:1')
        mean=torch.tensor([123.675, 116.28, 103.53])
        std=torch.tensor([58.395, 57.12, 57.375])
        

        rgb_img = torch.clone(img)
        rgb_img = rgb_img[:,[2,1,0],:,:]
        for i in range(3):
            rgb_img[:,i,:,:] = (rgb_img[:,i,:,:]-mean[i])/std[i]
        
        # Classic Faster R-CNN
        img_features = self.extract_feat(rgb_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                img_features,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            img_features, 
            img_metas, 
            proposal_list,
            gt_bboxes, 
            gt_labels,
            gt_bboxes_ignore, 
            gt_masks,
            **kwargs)
        losses.update(roi_losses)
        # end of classic Faster RCNN

        # shape of gt_nbboxes: list of batch length and each has: nbr bbox on image x nbr contrastive classes x [xmin ymin xmax ymax]
        # shape of gt_nlabels: list of batch length and each has: nbr bbox on image x nbr contrastive classes 
        
        # shape of nimg: list of batch length and each has: C x H x W
        # shape of img: list of batch length and each has: H x W x C
        rgb_img = torch.clone(img)
        rgb_img = rgb_img[:,[2,1,0],:,:]

        rgb_img = torch.movedim(rgb_img, (2,3), (1,2))


        img_aug, bboxes_aug, labels_aug, transform = self._nBBOX_augment(rgb_img, gt_nbboxes, gt_nlabels)

        # shape of img aug is batch x H x W x C
        img_aug = torch.cat((torch.unsqueeze(img_aug[0], dim=0), torch.unsqueeze(img_aug[1], dim=0)), dim=0)
        img_aug = torch.movedim(img_aug, (1, 2), (2,3))

        # shape of img aug is batch x C x H x W to match img
        c_aug_bboxes = []
        c_aug_labels = []
        gt_labels_aug_copy =[]


        for batch_id in range(len(bboxes_aug)):
            batch_reset_bbox = []
            batch_reset_lbl = []
            batch_reset_labels_aug_copy = []
            for bbox_number in range(len(bboxes_aug[batch_id])):
                for ccls_id in range(len(bboxes_aug[batch_id][bbox_number])):
                    batch_reset_bbox.append([
                        int(bboxes_aug[batch_id][bbox_number][ccls_id][0]),
                        int(bboxes_aug[batch_id][bbox_number][ccls_id][1]),
                        int(bboxes_aug[batch_id][bbox_number][ccls_id][2]),
                        int(bboxes_aug[batch_id][bbox_number][ccls_id][3])
                        ])
                    batch_reset_lbl.append(labels_aug[batch_id][bbox_number][ccls_id])
                    batch_reset_labels_aug_copy.append(int(gt_labels[batch_id][bbox_number]))

            gt_labels_aug_copy.append(torch.tensor(batch_reset_labels_aug_copy).to(device))
            c_aug_bboxes.append(torch.tensor(batch_reset_bbox).to(device))
            c_aug_labels.append(torch.tensor(batch_reset_lbl).to(device))
        gt_bboxes_aug = c_aug_bboxes
        gt_labels_aug = c_aug_labels
        
        # shape of aug gt bbox: list of batch length and each has: nbr bbox on image * nbr contrastive bbox x [xmin ymin xmax ymax]
        # shape of aug label: list of batch length and each has: nbr bbox on image * nbr contrastive classes 
        
        # normalize

        img_aug = img_aug.to(img.get_device())
        for i in range(3):
            img_aug[:,i,:,:] = (img_aug[:,i,:,:]-mean[i])/std[i]
        
        # Contrastive Faster R-CNN
        aug_img_features = self.extract_feat(img_aug)
        
        #for bbox_number in range(len(gt_bboxes)): 
        c_aug_labels = []
        c_aug_bboxes = []   
        for batch_id in range(len(gt_labels)):
            batch_lbl = []
            batch_bbox = []

            gt_nlabels[batch_id] = torch.flatten(gt_nlabels[batch_id], 0, 1)
            gt_nbboxes[batch_id] = torch.flatten(gt_nbboxes[batch_id], 0, 1)
            
            for bbox_id in range(gt_nlabels[batch_id].shape[0]):
                if gt_nlabels[batch_id][bbox_id] in gt_labels_aug[batch_id].to(gt_nlabels[batch_id][bbox_id].get_device()):
                    batch_lbl.append(gt_nlabels[batch_id][bbox_id].cpu().numpy())
                    batch_bbox.append(gt_nbboxes[batch_id][bbox_id].cpu().numpy())

            c_aug_labels.append(torch.tensor(np.stack(batch_lbl)).to(device))
            c_aug_bboxes.append(torch.tensor(np.stack(batch_bbox)).to(device))

        gt_nlabels = c_aug_labels
        gt_nbboxes = c_aug_bboxes
       # c_losses = dict()
    
        
        # TODO: flatten nlabels and nbbox to match sized of augmented and then check losses
        # RPN forward and loss
        if self.with_c_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            
            rpn_losses, proposal_list_c = self.c_rpn_head.forward_train(
                aug_img_features,
                img_metas,
                gt_bboxes_aug,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg,
                **kwargs)
            c_rpn_losses = dict()
            for key in rpn_losses.keys():
                c_rpn_losses['c_'+str(key)] = rpn_losses[key]
            losses.update(c_rpn_losses)
        else:
            proposal_list_c = proposals

        c_losses= True
        if c_losses:
            # Shapes:
            # gt_nbboxes, gt_bboxes_aug : batch x nbr c bbox x 4
            # gt_nlabels, gt_labels_aug : batch x nbr c bbox ¨

            roi_losses = self.c_roi_head.forward_train(
                img_features,
                img_metas,
                proposal_list,
                gt_nbboxes,
                gt_nlabels,
                aug_img_features,
                proposal_list_c,
                gt_bboxes_aug,
                gt_labels_aug,
                gt_labels_aug_copy,
                transform)
            losses.update(roi_losses)
            
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

@DETECTORS.register_module()
class OptimizeCFRCNN(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 c_rpn_head = None,
                 c_roi_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OptimizeCFRCNN, self).__init__(
            init_cfg=init_cfg)
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

        if c_rpn_head is not None:
            c_rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            c_rpn_head_ = c_rpn_head.copy()
            c_rpn_head_.update(train_cfg=c_rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.c_rpn_head = build_head(c_rpn_head_)

        if c_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            c_rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            c_roi_head.update(train_cfg=c_rcnn_train_cfg)
            c_roi_head.update(test_cfg=test_cfg.rcnn)
            c_roi_head.pretrained = pretrained
            self.c_roi_head = build_head(c_roi_head)
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
    
    @property
    def with_c_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.c_rpn_head is not None

    @property
    def with_c_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.c_roi_head is not None

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
    def forward(self, img, img_metas, base=None, novel=None, augmented=None, return_loss=True, **kwargs):
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
        # Classic Faster R-CNN
        losses = dict()

        img_features = self.extract_feat(img)
        

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                img_features,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            img_features, 
            img_metas, 
            proposal_list,
            gt_bboxes, 
            gt_labels,
            gt_bboxes_ignore, 
            gt_masks,
            **kwargs)
        losses.update(roi_losses)
        # end of classic Faster RCNN

        # shape of gt_nbboxes: list of batch length and each has: nbr bbox on image x nbr contrastive classes x [xmin ymin xmax ymax]
        # shape of gt_nlabels: list of batch length and each has: nbr bbox on image x nbr contrastive classes 
        
        # shape of nimg: list of batch length and each has: C x H x W
        # shape of img: list of batch length and each has: H x W x C

        # shape of img aug is batch x C x H x W to match img
        
        
        aug_img = augmented['img'].to(img.get_device())
        gt_nbboxes = novel['gt_bboxes']
        gt_nlabels = novel['gt_labels']
        aug_gt_nbboxes = augmented['gt_bboxes']
        aug_gt_nlabels = augmented['gt_labels']

        gt_nlabels_true = augmented['gt_labels_true'] #[torch.ones(aug_gt_nlabels[i].shape).to(gt_labels[i].get_device())*torch.unsqueeze(gt_labels[i], 0).T for i in range(len(gt_labels))]


        
        
        # shape of aug gt bbox: list of batch length and each has: nbr bbox on image * nbr contrastive bbox x [xmin ymin xmax ymax]
        # shape of aug label: list of batch length and each has: nbr bbox on image * nbr contrastive classes 
        
        # normalize

        
        # Contrastive Faster R-CNN
        aug_img_features = self.extract_feat(aug_img)
        
        #for bbox_number in range(len(gt_bboxes)): 
       
        
        # TODO: flatten nlabels and nbbox to match sized of augmented and then check losses
        # RPN forward and loss
        if self.with_c_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            
            rpn_losses, proposal_list_c = self.c_rpn_head.forward_train(
                aug_img_features,
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
            proposal_list_c = proposals

        c_losses= True
        if c_losses:
            # Shapes:
            # gt_nbboxes, gt_bboxes_aug : batch x nbr c bbox x 4
            # gt_nlabels, gt_labels_aug : batch x nbr c bbox ¨
            
            roi_losses = self.c_roi_head.forward_train(
                img_features,
                img_metas,
                proposal_list,
                gt_nbboxes,
                gt_nlabels,
                aug_img_features,
                proposal_list_c,
                aug_gt_nbboxes,
                aug_gt_nlabels,
                gt_nlabels_true)
            losses.update(roi_losses)
            
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

@DETECTORS.register_module()
class OptimizeCFRCNN_CIR(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 cir_neck = None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 c_rpn_head = None,
                 c_roi_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OptimizeCFRCNN, self).__init__(
            init_cfg=init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if cir_neck is not None:
            self.cir_neck = build_neck(cir_neck)

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

        if c_rpn_head is not None:
            c_rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            c_rpn_head_ = c_rpn_head.copy()
            c_rpn_head_.update(train_cfg=c_rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.c_rpn_head = build_head(c_rpn_head_)

        if c_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            c_rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            c_roi_head.update(train_cfg=c_rcnn_train_cfg)
            c_roi_head.update(test_cfg=test_cfg.rcnn)
            c_roi_head.pretrained = pretrained
            self.c_roi_head = build_head(c_roi_head)
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
    
    @property
    def with_c_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.c_rpn_head is not None

    @property
    def with_c_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.c_roi_head is not None

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
    
    def _nBBOX_augment(self, img, bboxes, labels):

        import albumentations as A

        transform = A.ReplayCompose(
                    [
                        A.HorizontalFlip(p=0.8),
                        A.VerticalFlip(p=0.8),
                        #A.Affine (scale=(0.9, 1.1), rotate=(0), p=0.8),
                       #A.RandomBrightnessContrast(p=0.3),
                    ],
                        bbox_params=
                            A.BboxParams(format='pascal_voc', min_visibility=0.5, label_fields=['class_labels']),
                    )
                    
        list_of_bbox = bboxes
        list_of_labels = labels
        
        aug_img = []
        aug_bboxes = []
        aug_labels = []
        # for each image
        for batch_id in range(len(list_of_bbox)):
            aug_bboxes_tmp = []
            aug_labels_tmp = []

            # for each object on the image
            for bbox_id in range(list_of_labels[batch_id].shape[0]):
                current_labels = list_of_labels[batch_id][bbox_id]
                
                current_bboxes = list_of_bbox[batch_id][bbox_id]
                
                current_img = img[batch_id].cpu().detach().numpy()
                current_labels = current_labels.cpu().detach().numpy()
                current_bboxes = current_bboxes.cpu().detach().numpy()
                
                if not bbox_id:
                    transformed = transform(image=current_img, bboxes=current_bboxes, class_labels=current_labels)
                    applied_transform = []
                    to_print = transformed['replay']['transforms']

                    for tran in transformed['replay']['transforms']:
                        if tran['applied']:
                            applied_transform.append([tran['__class_fullname__'], tran['params'], img[batch_id].shape])
                    
                else:
                    transformed = A.ReplayCompose.replay(transformed['replay'], image=current_img, bboxes=current_bboxes, class_labels=current_labels)
                    to_print = transformed['replay']['transforms']

                aug_bboxes_tmp.append(transformed['bboxes'])
                aug_labels_tmp.append(transformed['class_labels'])

            aug_bboxes.append(aug_bboxes_tmp)
            aug_labels.append(aug_labels_tmp)
            aug_img.append(torch.tensor(transformed['image']))

        return aug_img, aug_bboxes, aug_labels, applied_transform
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, base=None, novel=None, augmented=None, return_loss=True, **kwargs):
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
        # Classic Faster R-CNN
        img_features = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                img_features,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            img_features, 
            img_metas, 
            proposal_list,
            gt_bboxes, 
            gt_labels,
            gt_bboxes_ignore, 
            gt_masks,
            **kwargs)
        losses.update(roi_losses)
        # end of classic Faster RCNN
        #print(f'gt labels {gt_labels}, gt novel label {gt_nlabels}')
        #nimg = torch.cat((torch.unsqueeze(nimg[0], dim=0), torch.unsqueeze(nimg[1], dim=0)), dim=0)
        do_print = False
        if do_print:
            print(f'Shape of nimg {nimg.shape}')
            print(f'Shape of img {img.shape}')
            print(f'Len of nbbox {len(gt_nbboxes)}')
            for i in range(len(gt_nlabels)):
                print(f'shape of nbbox {gt_nbboxes[i].shape}')

            print(f'Len of nlabels {len(gt_nlabels)}')
            for i in range(len(gt_nlabels)):
                print(f'shape of nlabel {gt_nlabels[i].shape}')
                print(gt_nlabels[i])

            print(f'Len of bbox {len(gt_bboxes)}')
            for i in range(len(gt_nlabels)):
                print(f'shape of bbox {gt_bboxes[i].shape}')

            print(f'Len of labels {len(gt_labels)}')
            for i in range(len(gt_nlabels)):
                print(f'shape of label {gt_labels[i].shape}')

        # shape of gt_nbboxes: list of batch length and each has: nbr bbox on image x nbr contrastive classes x [xmin ymin xmax ymax]
        # shape of gt_nlabels: list of batch length and each has: nbr bbox on image x nbr contrastive classes 
        
        # shape of nimg: list of batch length and each has: C x H x W
        # shape of img: list of batch length and each has: H x W x C

        # shape of img aug is batch x C x H x W to match img
        
        
        aug_img = augmented['img'].to(img.get_device())
        gt_nbboxes = novel['gt_bboxes']
        gt_nlabels = novel['gt_labels']
        aug_gt_nbboxes = augmented['gt_bboxes']
        aug_gt_nlabels = augmented['gt_labels']

        gt_nlabels_true = augmented['gt_labels_true'] #[torch.ones(aug_gt_nlabels[i].shape).to(gt_labels[i].get_device())*torch.unsqueeze(gt_labels[i], 0).T for i in range(len(gt_labels))]


        do_print = False
        if do_print:
            print(f' gt labels {gt_labels}')
            print(f' gt nlabels {gt_nlabels}')
            print(f' gt aug labels {aug_gt_nlabels}')
            print(f' true labe {gt_nlabels_true}')
            print(f'Shape of aug img {aug_img.shape}')
            print(f'Shape of  img {img.shape}')
            print(f'Len of aug bbox {len(aug_gt_nbboxes)}')
            for i in range(len(aug_gt_nbboxes)):
                print(f'shape of aug bbox {aug_gt_nbboxes[i].shape}')

            print(f'Len of bbox {len(gt_nbboxes)}')
            for i in range(len(gt_nbboxes)):
                print(f'shape of aug bbox {gt_nbboxes[i].shape}')

            print(f'Len of bbox {len(gt_bboxes)}')
            for i in range(len(gt_bboxes)):
                print(f'shape of aug bbox {gt_bboxes[i].shape}')

        
        
        # shape of aug gt bbox: list of batch length and each has: nbr bbox on image * nbr contrastive bbox x [xmin ymin xmax ymax]
        # shape of aug label: list of batch length and each has: nbr bbox on image * nbr contrastive classes 
        
        # normalize

        
        # Contrastive Faster R-CNN
        aug_img_features = self.extract_feat(aug_img)
        
        #for bbox_number in range(len(gt_bboxes)): 
       
        
        # TODO: flatten nlabels and nbbox to match sized of augmented and then check losses
        # RPN forward and loss
        if self.with_c_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            
            rpn_losses, proposal_list_c = self.c_rpn_head.forward_train(
                aug_img_features,
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
            proposal_list_c = proposals

        c_losses= True
        if c_losses:
            # Shapes:
            # gt_nbboxes, gt_bboxes_aug : batch x nbr c bbox x 4
            # gt_nlabels, gt_labels_aug : batch x nbr c bbox ¨
            
            roi_losses = self.c_roi_head.forward_train(
                img_features,
                img_metas,
                proposal_list,
                gt_nbboxes,
                gt_nlabels,
                aug_img_features,
                proposal_list_c,
                aug_gt_nbboxes,
                aug_gt_nlabels,
                gt_nlabels_true)
            losses.update(roi_losses)
            
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

@DETECTORS.register_module()
class OptimizeCosSimCFRCNN(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 c_rpn_head = None,
                 c_roi_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OptimizeCosSimCFRCNN, self).__init__(
            init_cfg=init_cfg)
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

        if c_rpn_head is not None:
            c_rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            c_rpn_head_ = c_rpn_head.copy()
            c_rpn_head_.update(train_cfg=c_rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.c_rpn_head = build_head(c_rpn_head_)

        if c_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            c_rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            c_roi_head.update(train_cfg=c_rcnn_train_cfg)
            c_roi_head.update(test_cfg=test_cfg.rcnn)
            c_roi_head.pretrained = pretrained
            self.c_roi_head = build_head(c_roi_head)
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
    
    @property
    def with_c_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.c_rpn_head is not None

    @property
    def with_c_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.c_roi_head is not None

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
    def forward(self, img, img_metas, base=None, novel=None, augmented=None, return_loss=True, **kwargs):
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
        # Classic Faster R-CNN
        losses = dict()

        img_features = self.extract_feat(img)
        

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                img_features,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            img_features, 
            img_metas, 
            proposal_list,
            gt_bboxes, 
            gt_labels,
            gt_bboxes_ignore, 
            gt_masks,
            **kwargs)
        losses.update(roi_losses)
        # end of classic Faster RCNN




        # cosine sim of nbboxes and bboxes

        
        #batch, channel, height, width = all_features.size(0), all_features.size(1), all_features.size(2), all_features.size(3)
        #all_features_reshape = all_features.view(batch, -1).contiguous()

        # cosine similarity
        #dot_product_mat = torch.matmul(all_features_reshape, torch.transpose(all_features_reshape, 0, 1))
        #len_vec = torch.unsqueeze(torch.sqrt(torch.sum(all_features_reshape * all_features_reshape, dim=1)), dim=0)
        #len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        #cos_sim_mat = dot_product_mat / len_mat / batch

        #all_features_new = torch.mm(cos_sim_mat, all_features_reshape).view(batch, channel, height, width)+all_features


        # shape of gt_nbboxes: list of batch length and each has: nbr bbox on image x nbr contrastive classes x [xmin ymin xmax ymax]
        # shape of gt_nlabels: list of batch length and each has: nbr bbox on image x nbr contrastive classes 
        
        # shape of nimg: list of batch length and each has: C x H x W
        # shape of img: list of batch length and each has: H x W x C

        # shape of img aug is batch x C x H x W to match img
        
        
        aug_img = augmented['img'].to(img.get_device())
        gt_nbboxes = novel['gt_bboxes']
        gt_nlabels = novel['gt_labels']
        aug_gt_nbboxes = augmented['gt_bboxes']
        aug_gt_nlabels = augmented['gt_labels']

        gt_nlabels_true = augmented['gt_labels_true'] #[torch.ones(aug_gt_nlabels[i].shape).to(gt_labels[i].get_device())*torch.unsqueeze(gt_labels[i], 0).T for i in range(len(gt_labels))]


        
        
        # shape of aug gt bbox: list of batch length and each has: nbr bbox on image * nbr contrastive bbox x [xmin ymin xmax ymax]
        # shape of aug label: list of batch length and each has: nbr bbox on image * nbr contrastive classes 
        
        # normalize

        
        # Contrastive Faster R-CNN
        aug_img_features = self.extract_feat(aug_img)
        
        #for bbox_number in range(len(gt_bboxes)): 
       
        
        # TODO: flatten nlabels and nbbox to match sized of augmented and then check losses
        # RPN forward and loss
        if self.with_c_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            
            rpn_losses, proposal_list_c = self.c_rpn_head.forward_train(
                aug_img_features,
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
            proposal_list_c = proposals

        c_losses= True
        if c_losses:
            # Shapes:
            # gt_nbboxes, gt_bboxes_aug : batch x nbr c bbox x 4
            # gt_nlabels, gt_labels_aug : batch x nbr c bbox ¨
            
            roi_losses = self.c_roi_head.forward_train(
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
                self.roi_head.bbox_results_cosine,
                self.roi_head.bbox_targets_cosine)
            losses.update(roi_losses)
            
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

@DETECTORS.register_module()
class CosSimFRCNN(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CosSimFRCNN, self).__init__(
            init_cfg=init_cfg)
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
    def forward(self, img, img_metas, base=None, novel=None, augmented=None, return_loss=True, **kwargs):
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

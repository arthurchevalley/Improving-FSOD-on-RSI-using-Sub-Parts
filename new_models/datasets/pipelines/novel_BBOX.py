from ..builder import PIPELINES
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import save_image
import torch
import torchvision
from .transforms import Albu
from mmcv.parallel import DataContainer as DC
import albumentations as A
import torch
import torch.nn as nn

import mmcv
import torchvision
from collections.abc import Sequence

import sys
import math

import math
from skimage import io, color
from tqdm import trange



def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {
        'gt_bboxes': 'gt_labels',
        'gt_bboxes_ignore': 'gt_labels_ignore'
    }
    bbox2mask = {
        'gt_bboxes': 'gt_masks',
        'gt_bboxes_ignore': 'gt_masks_ignore'
    }
    bbox2seg = {
        'gt_bboxes': 'gt_semantic_seg',
    }
    return bbox2label, bbox2mask, bbox2seg


@PIPELINES.register_module()
class AugScaled_nBBOX:
    """
    Sub-Parts main classes that handles creation and augmentation

    Args:
        out_max; Define the maximum number of pixels from the Sub-Parts that geos out of the base object
        nbr_nBBOX; Define the number of Sub-Parts
        min_bbox_size; Define the minium box width/height of the Sub-Parts
        batch_size; Dataloader batch size 
        nbr_class; Number of classes in the dataset
        BBOX_scaling; defines the minimum Sub-Parts size based on the ratio of original object 
    """

    def __init__(self,
                 out_max = 15,
                 nbr_nBBOX = 3,
                 min_bbox_size = 5,
                 batch_size = 2,
                 nbr_class = 20,
                 BBOX_scaling = 0.3
                 ):
                 
        self.nbr_nBBOX = nbr_nBBOX
        self.out_max = out_max
        self.start_id = nbr_class + 1
        self.batch_size = batch_size
        self.t = 0
        self.reset_cntr = 0
        self.BBOX_scaling = BBOX_scaling
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        
    def _create_nBBOX_Scaled(self, results):

        if np.any(results['img']<0):
            print("Neg before create nBBOX")
        bboxes = results['gt_bboxes'] 
        nbboxes_per_bbox = []
        nlabels_per_bbox = []
        gt_nlabel_true = []
        img_shape = results['img_shape']
        gt_nlabel = []
        gt_nbboxes = []

        for id in range(bboxes.shape[0]):
            xmin = int(bboxes[id][0])
            ymin = int(bboxes[id][1])
            xmax = int(bboxes[id][2])
            ymax = int(bboxes[id][3])

            # Only keep the valide BBOX
            if xmin >= xmax or ymin >= ymax: 
                continue
            for cid in range(self.nbr_nBBOX):
                # Random center selection
                cx = np.random.randint(xmin, xmax)
                cy = np.random.randint(ymin, ymax)
                scalex, scaley = np.random.random(2)

                # Defines the Sub-Part width and height
                dx = scalex*max(cx - xmin, xmax - cx)
                dy = scaley*max(cy - ymin, ymax - cy)
                BBOX_scale_X, BBOX_scale_Y = self.BBOX_scaling + (1-self.BBOX_scaling) * np.random.random(2)
                if 2*dx < BBOX_scale_X*(xmax - xmin):
                    dx = BBOX_scale_X*(xmax - xmin)/2
                if 2*dy < BBOX_scale_Y*(ymax - ymin):
                    dy = BBOX_scale_Y*(xmax - xmin)/2
                nxmin = int(cx - dx)
                nxmax = int(cx + dx)
                nymin = int(cy - dy)
                nymax = int(cy + dy)

                # Handles the maximum Sub-Part overflow of the base object
                if self.out_max: 
                    if nxmin < (xmin - self.out_max): nxmin = xmin - self.out_max
                    if nxmax > (xmax + self.out_max): nxmax = xmax + self.out_max
                    if nymin < (ymin - self.out_max): nymin = ymin - self.out_max
                    if nymax > (ymax + self.out_max): nymax = ymax + self.out_max
                else:
                    if nxmin < xmin: nxmin = xmin
                    if nxmax > xmax: nxmax = xmax
                    if nymin < ymin: nymin = ymin
                    if nymax > ymax: nymax = ymax
                nxmin = max(0, nxmin)
                nymin = max(0, nymin)
                nxmax = min(img_shape[0], nxmax)
                nymax = min(img_shape[1], nymax)
                gt_nbboxes.append([nxmin, nymin, nxmax, nymax])
                gt_nlabel.append(self.start_id + cid + self.t)
                gt_nlabel_true.append(results['gt_labels'][id])

            self.t += self.nbr_nBBOX

        nbboxes_per_bbox = np.array(gt_nbboxes)
        nlabels_per_bbox = np.array(gt_nlabel)
        gt_nlabel_true = np.array(gt_nlabel_true)
        results['gt_bboxes'] = nbboxes_per_bbox
        results['gt_labels'] = nlabels_per_bbox
        results['gt_labels_true'] = gt_nlabel_true
        
        self.reset_cntr += 1
        if not self.reset_cntr%self.batch_size:
            self.t = 0


    def _nBBOX_augment(self, results):
        """
        Augmentations of the original BBOX as well as the Sub-Parts
        """
        img = results['nimg']
        bboxes = results['gt_nbboxes']
        labels = results['gt_nlabels']

        transform = A.ReplayCompose(
                    [
                        A.HorizontalFlip(p=0.8),
                        A.VerticalFlip(p=0.8),
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
                
                current_img = img[batch_id]
                
                
                if not bbox_id:
                    transformed = transform(image=current_img, bboxes=current_bboxes, class_labels=current_labels)
                    applied_transform = []
                    to_print = transformed['replay']['transforms']

                    for tran in transformed['replay']['transforms']:
                        if tran['applied']:
                            applied_transform.append([tran['__class_fullname__'], tran['params']])
                    
                else:
                    transformed = A.ReplayCompose.replay(transformed['replay'], image=current_img, bboxes=current_bboxes, class_labels=current_labels)
                    to_print = transformed['replay']['transforms']

                aug_bboxes_tmp.append(transformed['bboxes'])
                aug_labels_tmp.append(transformed['class_labels'])

            aug_bboxes.append(aug_bboxes_tmp)
            aug_labels.append(aug_labels_tmp)
            aug_img.append(torch.tensor(transformed['image']))

        img_aug = torch.cat((torch.unsqueeze(img_aug[0], dim=0), torch.unsqueeze(img_aug[1], dim=0)), dim=0)
        img_aug = torch.movedim(img_aug, (1, 2), (2,3))

        results['aug_img'] = aug_img
        results['aug_bboxes'] = aug_bboxes
        results['aug_labels'] = aug_labels
        results['applied_transform'] = applied_transform
        

    def __call__(self, results):
        self._create_nBBOX_Scaled(results)
        
        return results


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@PIPELINES.register_module()
class ContrastiveCollect:
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results.keys():
                img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            if key in results.keys():
                data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'

  
@PIPELINES.register_module()
class NimgContrastiveDefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 to_format = ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels'],
                 pad_val=dict(img=0, nimg=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val
        self.keys = to_format

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results and 'nimg' in results:

            for img_key in ['img', 'nimg']:
                img = results[img_key]
                if self.img_to_float is True and img.dtype == np.uint8:
                    # Normally, image is of uint8 type without normalization.
                    # At this time, it needs to be forced to be converted to
                    # flot32, otherwise the model training and inference
                    # will be wrong. Only used for YOLOX currently .
                    img = img.astype(np.float32)
                # add default meta keys
                results = self._add_default_meta_keys(results, img_key)
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results[img_key] = DC(
                    to_tensor(img), padding_value=self.pad_val[img_key], stack=True)
        for key in self.keys:
            if key not in results or key == 'applied_transformation':
                continue
            results[key] = DC(
                to_tensor(results[key]),
                stack=False
                )
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results, img_key):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        
        img = results[img_key]
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'

@PIPELINES.register_module()
class ContrastiveRandomFlip:
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        
        if 'contrastive_flip_'+str(self.direction) not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            #results['contrastive_flip_'+str(self.direction) ] = cur_dir is not None
            

        #if 'contrastive_flip_direction_'+str(self.direction) not in results:
        #    results['contrastive_flip_direction_'+str(self.direction)] = cur_dir
        #if results['contrastive_flip_'+str(self.direction)]:
        if cur_dir is not None:
            # flip image
            key = 'nimg'
            results[key] = mmcv.imflip(
                    results[key], direction=self.direction)
            # flip bboxes
            key = 'gt_nbboxes'
            results['aug_'+str(key)] = self.bbox_flip(results[key],
                                            results['img_shape'],
                                            self.direction)
            if 'applied_transformation' in results:
                new_transforamtion = [[self.direction], results['img_shape']] + results['applied_transformation']
                results.update(applied_transformation=new_transforamtion)
            else:
                results['applied_transformation'] = [[self.direction], results['img_shape']]
                                            
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'

@PIPELINES.register_module()
class MultiRandomFlip:
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        
        if 'contrastive_flip_'+str(self.direction) not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            #results['contrastive_flip_'+str(self.direction) ] = cur_dir is not None
            

        #if 'contrastive_flip_direction_'+str(self.direction) not in results:
        #    results['contrastive_flip_direction_'+str(self.direction)] = cur_dir
        #if results['contrastive_flip_'+str(self.direction)]:
        if cur_dir is not None:
            # flip image

            results['img'] = mmcv.imflip(
                    results['img'], direction=self.direction)
            # flip bboxes

            results.update(gt_bboxes=self.bbox_flip(results['gt_bboxes'],
                                            results['img_shape'],
                                            self.direction))
            if 'applied_transformation' in results:
                new_transforamtion = [self.direction] + results['applied_transformation']
                results.update(applied_transformation=new_transforamtion)
            else:
                results['applied_transformation'] = [self.direction]
                                            
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'

@PIPELINES.register_module()
class NovelNormalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        
        for key in ['nimg', 'img']:
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
            
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ContrastiveDefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 to_format = ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_labels_true'],
                 pad_val=dict(img=0, nimg=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val
        self.keys = to_format

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        for img_key in ['img']:
            img = results[img_key]
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results, img_key)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results[img_key] = DC(
                to_tensor(img), padding_value=self.pad_val[img_key], stack=True)
        for key in self.keys:
            if key not in results:
                continue
            results[key] = DC(
                to_tensor(results[key]),
                stack=False
                )
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results, img_key):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        
        img = results[img_key]
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'


@PIPELINES.register_module()
class ContrastiveRotate:
    """Apply Rotate Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``mmcv.imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        rotate_angle (int | float): The maximum angles for rotate
            transformation.
    """

    def __init__(self,
                 center=None,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 rotate_angle=[90, 180]):
                 
        if isinstance(center, (int, float)):
            center = (center, center)
        elif isinstance(center, tuple):
            assert len(center) == 2, 'center with type tuple must have '\
                f'2 elements. got {len(center)} elements.'
        else:
            assert center is None, 'center must be None or type int, '\
                f'float or tuple, got type {type(center)}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must '\
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255]. '\
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            f'got {prob}.'
        assert isinstance(rotate_angle, (int, float, list)), 'rotate_angle '\
            f'should be type int or float. got type {type(rotate_angle)}.'

        # Rotation angle in degrees. Positive values mean
        # clockwise rotation.
        self.angle = rotate_angle
        self.center = center
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.rotate_angle = rotate_angle

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob:
            return results
        h, w = results['img'].shape[:2]
        angle = int(np.random.choice(self.rotate_angle,1))

        angle = -angle if np.random.rand() <= 0.5 else angle
        
        img = torch.from_numpy(results['img'].copy())
        img = img.movedim(-1,0)

        img_rotated = torchvision.transforms.functional.rotate(img, angle)

        img_rotated = img_rotated.movedim(0, -1)
        
        results['img'] = img_rotated.numpy()
        bboxes = results['gt_bboxes']

        if angle > 0:
            # CCW
            xmin, ymin, xmax, ymax = results['gt_bboxes'][:,1], w-results['gt_bboxes'][:,2], results['gt_bboxes'][:,3], w-results['gt_bboxes'][:,0]
            if angle > 90:
                xmin, ymin, xmax, ymax = ymin, w-xmax, ymax, w-xmin
        else:
            #CW
            xmin, ymin, xmax, ymax = h-results['gt_bboxes'][:,3], results['gt_bboxes'][:,0], h-results['gt_bboxes'][:,1], results['gt_bboxes'][:,2]
            if angle < 90:
                xmin, ymin, xmax, ymax = h-ymax, xmin, h-ymin, xmax
        results['gt_bboxes'] = np.stack((xmin,ymin,xmax,ymax)).T
        
        if 'applied_transformation' in results:
                new_transforamtion = [str(angle)] + results['applied_transformation']
                results.update(applied_transformation=new_transforamtion)
        else:
            results['applied_transformation'] = [str(angle)]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'center={self.center}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'rotate_angle={self.rotate_angle}, '
        return repr_str

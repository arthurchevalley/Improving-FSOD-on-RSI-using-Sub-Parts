# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial

from typing import Dict, List, Optional, Sequence, Union
import mmcv
import numpy as np


from mmcv.utils import print_log
from mmdet.core import eval_recalls,eval_map

import torch
from mmcv.ops import nms_rotated

import xml.etree.ElementTree as ET
from collections import OrderedDict

from .builder import DATASETS
from .base import BaseFewShotDataset
from mmdet.datasets.custom import CustomDataset

DOTA_SPLIT = dict(
    ALL_CLASSES_SPLIT1 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane'),

    ALL_CLASSES_SPLIT3 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                            'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 
                            'harbor', 'swimming-pool', 'helicopter', 'container-crane'),

    NOVEL_CLASSES_SPLIT3 = ('plane',  'storage-tank', 'soccer-ball-field', 'roundabout', 
                            'swimming-pool', 'container-crane'),

    BASE_CLASSES_SPLIT3 = ('baseball-diamond', 'bridge', 'ground-track-field',
                            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                            'basketball-court', 'harbor', 'helicopter'),


    ALL_CLASSES_SPLIT2 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
                'container-crane','helipad'),
    NOVEL_CLASSES_SPLIT1 = (['container-crane']),

    NOVEL_CLASSES_SPLIT2 = ('container-crane','helipad'),
    BASE_CLASSES_SPLIT1 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                            'basketball-court', 'storage-tank', 'soccer-ball-field','roundabout', 
                            'harbor', 'swimming-pool', 'helicopter'))

DOTA_PALETTE = dict(
    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255),
               (128, 128, 0), (0, 128, 128), (128, 0, 128)],

    PALETTE_NOVEL1 = [(128, 128, 0)],

    PALETTE_NOVEL2 = [(128, 128, 0), (0, 128, 128)],

    PALETTE_NOVEL3 = [(128, 128, 0), (0, 128, 128), (128, 0, 128)],

    PALETTE_BASE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
                    (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
                    (255, 255, 0), (147, 116, 116), (0, 0, 255)])


@DATASETS.register_module()
class FewShotDotaDatasetHBB(BaseFewShotDataset):
    """VOC dataset for few shot detection.

    Args:
        classes (str | Sequence[str]): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load pre-defined classes in `FewShotVOCDataset`.
            For example: 'NOVEL_CLASSES_SPLIT1'.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used
            for each base class. When it is None, all annotations
            will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
            Default: None.
        use_difficult (bool): Whether use the difficult annotation or not.
            Default: False.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
            this filter. Default: None.
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
        coordinate_offset (list[int]): The bbox annotation will add the
            coordinate offsets which corresponds to [x_min, y_min, x_max,
            y_max] during training. For testing, the gt annotation will
            not be changed while the predict results will minus the
            coordinate offsets to inverse data loading logic in training.
            Default: [-1, -1, 0, 0].
    """

    def __init__(self,
                 save_dataset = False,
                classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict] = None,
                 use_difficult: bool = False,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 dataset_name: Optional[str] = None,
                 test_mode: bool = False,
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 version = 'oc',
                 fsod_enable = False,
                 difficulty = 100,
                 **kwargs) -> None:
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        self.SPLIT = DOTA_SPLIT
        self.version = version
        self.difficulty = difficulty
        # the split_id would be set value in `self.get_classes`
        self.split_id = None

        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotDotaDataset` can not be None.'
        
        self.fsod = fsod_enable
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.CLASSES = self.get_classes(classes)
        # `ann_shot_filter` will be used to filter out excess annotations
        # for few shot setting. It can be configured manually or generated
        # by the `num_novel_shots` and `num_base_shots`
        if ann_shot_filter is None:
            # configure ann_shot_filter by num_novel_shots and num_base_shots
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'
        self.coordinate_offset = coordinate_offset
        self.use_difficult = use_difficult
        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        #
        #print(img_info['filename'])
        id = img_info['filename']
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        #print(results['ann_info']['bboxes'])
        
        if torch.isnan(torch.from_numpy(results['ann_info']['bboxes'])).any():
            print(f'nan before for image {id}')
        to_return = self.pipeline(results)
        #if to_return['gt_bboxes'] is not None:
        #        print(f'nan before for image {id}')
        ##    if torch.isnan(to_return['gt_bboxes']._data).any():
        
        return to_return

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES_SPLIT1', 'ALL_CLASSES_SPLIT2', 'ALL_CLASSES_SPLIT3',
         'BASE_CLASSES_SPLIT1', 'BASE_CLASSES_SPLIT2', 'BASE_CLASSES_SPLIT3',
         'NOVEL_CLASSES_SPLIT1','NOVEL_CLASSES_SPLIT2','NOVEL_CLASSES_SPLIT3']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotVOCDataset`.
                For example: 'NOVEL_CLASSES_SPLIT1'.

        Returns:
            list[str]: List of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name}: not a pre-defined classes or ' \
               f'split in DOTA_SPLIT'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
            self.split_id = int(classes[-1])
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self) -> Dict[str, int]:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT[
                    f'NOVEL_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT[f'BASE_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def load_annotations(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Support to load annotation from two type of ann_cfg.

        Args:
            ann_cfg (list[dict]): Support two type of config.

            - loading annotation from common ann_file of dataset
              with or without specific classes.
              example:dict(type='ann_file', ann_file='path/to/ann_file',
              ann_classes=['dog', 'cat'])
            - loading annotation from a json file saved by dataset.
              example:dict(type='saved_dataset', ann_file='path/to/ann_file')

        Returns:
            list[dict]: Annotation information.
        """

        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        for ann_cfg_ in ann_cfg:
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            elif ann_cfg_['type'] == 'ann_file':
                # load annotation from specific classes
                ann_classes = ann_cfg_.get('ann_classes', None)
                if ann_classes is not None:
                    for c in ann_classes:
                        assert c in self.CLASSES, \
                            f'{self.dataset_name}: ann_classes must in ' \
                            f'dataset classes.'
                else:
                    ann_classes = self.CLASSES
                data_infos += self.load_annotations_xml(
                    ann_cfg_['ann_file'], ann_classes)
            else:
                raise ValueError(
                    f'{self.dataset_name}: not support '
                    f'annotation type {ann_cfg_["type"]} in ann_cfg.')
        return data_infos

    def load_annotations_xml(
            self,
            ann_file: str,
            classes: Optional[List[str]] = None) -> List[Dict]:
        """Load annotation from XML style ann_file.

        It supports using image id or image path as image names
        to load the annotation file.

        Args:
            ann_file (str): Path of annotation file.
            classes (list[str] | None): Specific classes to load form xml file.
                If set to None, it will use classes of whole dataset.
                Default: None.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_names = mmcv.list_from_file(ann_file)

        for img_name in img_names:
            # ann file in image path format
            if 'dota' in ann_file:
                if 'val' in ann_file:
                    img_id = img_name
                else:
                    img_id = img_name

                if self.fsod and 'val' not in ann_file:
                    img_id = img_name[7:-4]
                else:
                    img_id = img_name
                filename = f'{img_id}.png'

            else:
                raise ValueError('Cannot infer dataset year from img_prefix')
            if 'dota' in ann_file:

                if 'val' in ann_file:
                    ann_info, width, height = self._get_xml_ann_info_dota_hbb(img_id, classes, True)
                else:
                    ann_info, width, height = self._get_xml_ann_info_dota_hbb(img_id, classes)

            data_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann = ann_info))

        return data_infos

    def _get_xml_ann_info_dota_hbb(self,
                          img_id: str,
                          classes: Optional[List[str]] = None,
                          validation = False) -> Dict:
        """Get annotation from XML file by img_id.

        Args:
            img_id (str): Id of image.
            classes (list[str] | None): Specific classes to load form
                xml file. If set to None, it will use classes of whole
                dataset. Default: None.

        Returns:
            dict: Annotation info of specified id with specified class.
        """
        if classes is None:
            classes = self.CLASSES
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        if validation:
            xml_path = f'/home/data/dota-1.5/val/labelTxt-v1.5/DOTA-v1.5_val_hbb/xml/{img_id}.xml'
        else:
            xml_path = f'/home/data/dota-1.5/train/labelTxt-v1.5/DOTA-v1.5_train_hbb/xml/{img_id}.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in classes:
                continue
            label = self.cat2label[name]
            if self.use_difficult:
                difficult = 0
            else:
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')

            # It should be noted that in the original mmdet implementation,
            # the four coordinates are reduced by 1 when the annotation
            # is parsed. Here we following detectron2, only xmin and ymin
            # will be reduced by 1 during training. The groundtruth used for
            # evaluation or testing keep consistent with original xml
            # annotation file and the xmin and ymin of prediction results
            # will add 1 for inverse of data loading logic.
            # x1 = x4 = xmin, x2 = x3 = xmax, y1 = y2 = ymin, y3 = y4 = ymax from OBB to HBB
            # bbox info xmin ymin xmax ymax

            xmin = int(float(bnd_box.find('x1').text))
            ymin = int(float(bnd_box.find('y1').text))
            xmax = int(float(bnd_box.find('x3').text))
            ymax = int(float(bnd_box.find('y3').text))

            bbox_info = [
                xmin,
                ymin,
                xmax,
                ymax
            ]
            #print(bbox_info)
            ignore = False
            if self.min_bbox_area:
                assert not self.test_mode
                w = bbox_info[2] - bbox_info[0]
                h = bbox_info[3] - bbox_info[1]
                if (w*h) < self.min_bbox_area:
                    ignore = True
            if difficult or ignore:
                #print('in ignore')
                bboxes_ignore.append(bbox_info)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox_info)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2)-1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)-1
            labels_ignore = np.array(labels_ignore)
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info, width, height

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images not meet the demand.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of `data_infos`.
        """
        valid_inds = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt:
                cat_ids = img_info['ann']['labels'].astype(np.int64).tolist()
                if len(cat_ids) == 0:
                    continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
        return valid_inds

    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'mAP',
                 logger: Optional[object] = None,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thr: Optional[Union[float, Sequence[float]]] = 0.5,
                 class_splits: Optional[List[str]] = None) -> Dict:
        """Evaluation in VOC protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Predictions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in VOC_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        
        

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                print(self.CLASSES)
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    #classes = self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                # calculate evaluate results of different class splits
                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if class_splits is not None:
                for k in class_splits.keys():
                    mAP = sum(class_splits_mean_aps[k]) / len(
                        class_splits_mean_aps[k])
                    print_log(f'{k} mAP: {mAP}', logger=logger)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, 
                results, 
                proposal_nums,
                iou_thr, 
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results


@DATASETS.register_module()
class FewShotDotaDefaultDatasetHBB(FewShotDotaDatasetHBB):
    """Dataset with some pre-defined VOC annotation paths.

    :obj:`FewShotVOCDefaultDataset` provides pre-defined annotation files
    to ensure the reproducibility. The pre-defined annotation files provide
    fixed training data to avoid random sampling. The usage of `ann_cfg' is
    different from :obj:`FewShotVOCDataset`. The `ann_cfg' should contain
    two filed: `method` and `setting`.

    Args:
        ann_cfg (list[dict]): Each dict should contain
            `method` and `setting` to get corresponding
            annotation from `DEFAULT_ANN_CONFIG`.
            For example: [dict(method='TFA', setting='SPILT1_1shot')].
    """

    dota_benchmark = {
        f'SPLIT{split}_{shot}SHOT': [
            dict(
                type='ann_file',
                ann_file=f'/home/data/dota-1.5/few_shot_ann/benchmark_{shot}shot/'
                f'xView_box_{shot}shot_{class_name}_train.txt',
                ann_classes=[class_name])
            for class_name in DOTA_SPLIT[f'ALL_CLASSES_SPLIT{split}']
        ]
        for shot in [1, 5, 10, 100] for split in [1, 2, 3]
    }

    # pre-defined annotation config for model reproducibility
    DEFAULT_ANN_CONFIG = dict(
        TFA=dota_benchmark)

    def __init__(self, ann_cfg: List[Dict], **kwargs) -> None:
        self.fsod = True
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Parse pre-defined annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): contain method and setting
                of pre-defined annotation config. Example:
                [dict(method='TFA', setting='SPILT1_1shot')]

        Returns:
            list[dict]: Annotation information.
        """
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name}: ann_cfg should be list of dict.'
            method = ann_cfg_['method']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[method][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = osp.join(
                        ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        return super(FewShotDotaDefaultDatasetHBB, self).ann_cfg_parser(new_ann_cfg)
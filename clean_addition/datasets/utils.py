# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmcv.cnn import VGG
from mmcv.runner.hooks import HOOKS, Hook

from mmdet.datasets.pipelines import (LoadAnnotations, LoadImageFromFile,
                                      LoadPanopticAnnotations)
from mmdet.models.dense_heads import GARPNHead, RPNHead
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


def get_loading_pipeline(pipeline):
    """Only keep loading image and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True),
        ...    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        ...    dict(type='RandomFlip', flip_ratio=0.5),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle'),
        ...    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True)
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        obj_cls = PIPELINES.get(cfg['type'])
        # TODO：use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (LoadImageFromFile,
                                               LoadAnnotations,
                                               LoadPanopticAnnotations):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg



# Copyright (c) OpenMMLab. All rights reserved.
import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Save numpy array obj to json."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_copy_dataset_type(dataset_type: str) -> str:
    """Return corresponding copy dataset type."""
    if dataset_type in ['FewShotVOCDataset', 'FewShotVOCDefaultDataset']:
        copy_dataset_type = 'FewShotVOCCopyDataset'
    elif dataset_type in ['FewShotCocoDataset', 'FewShotCocoDefaultDataset']:
        copy_dataset_type = 'FewShotCocoCopyDataset'
    elif dataset_type in ['FewShotDiorDataset', 'FewShotDiorDefaultDataset']:
        copy_dataset_type = 'FewShotDiorCopyDataset'
    elif dataset_type in ['FewShotDiorDefaultDatasetNoDif']:
        copy_dataset_type = 'FewShotDiorCopyDefaultDatasetNoDif'
    elif dataset_type in ['ContrastiveFewShotDiorDefaultDatasetNoDif']:
        copy_dataset_type = 'ContrastiveFewShotDiorCopyDefaultDatasetNoDif'
    else:
        raise TypeError(f'{dataset_type} '
                        f'not support copy data_infos operation.')

    return copy_dataset_type

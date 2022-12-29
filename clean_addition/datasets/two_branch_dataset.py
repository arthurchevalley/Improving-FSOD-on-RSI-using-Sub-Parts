# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import collections
import copy
import math
import json

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

import numpy as np
from mmcv.utils import build_from_cfg, print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS, PIPELINES

from .base import BaseFewShotDataset
from .dior import FewShotDiorDataset


@DATASETS.register_module()
class TwoBranchDataset:
    """A dataset wrapper of TwoBranchDataset.
    Wrapping main_dataset and auxiliary_dataset to a single dataset and thus
    building TwoBranchDataset requires two dataset. The behavior of
    TwoBranchDataset is determined by `mode`. Dataset will return images
    and annotations according to `mode`, e.g. fetching data from
    main_dataset if `mode` is 'main'. The default `mode` is 'main' and
    by using convert function `convert_main_to_auxiliary` the `mode`
    will be converted into 'auxiliary'.
    Args:
        main_dataset (:obj:`FewShotDiorDataset`):
            Main dataset to be wrapped.
        auxiliary_dataset (:obj:`FewShotDiorDataset` | None):
            Auxiliary dataset to be wrapped. If auxiliary dataset is None,
            auxiliary dataset will copy from main dataset.
        reweight_dataset (bool): Whether to change the sampling weights
            of VOC07 and VOC12 . Default: False.
    """

    def __init__(self,
                 main_dataset = None,
                 auxiliary_dataset = None,
                 reweight_dataset = False):
        assert main_dataset and auxiliary_dataset
        self._mode = 'main'
        self.main_dataset = main_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.CLASSES = self.main_dataset.CLASSES
        if reweight_dataset:
            # Reweight the VOC dataset to be consistent with the original
            # implementation of MPSR. For more details, please refer to
            # https://github.com/jiaxi-wu/MPSR/blob/master/maskrcnn_benchmark/data/datasets/voc.py#L137
            self.main_idx_map = self.reweight_dataset(
                self.main_dataset,
                ['VOC2007', 'VOC2012'],
            )
            self.auxiliary_idx_map = self.reweight_dataset(
                self.auxiliary_dataset, ['VOC'])
        else:
            self.main_idx_map = list(range(len(self.main_dataset)))
            self.auxiliary_idx_map = list(range(len(self.auxiliary_dataset)))
        self._main_len = len(self.main_idx_map)
        self._auxiliary_len = len(self.auxiliary_idx_map)
        self._set_group_flag()

    def __getitem__(self, idx: int) -> Dict:
        if self._mode == 'main':
            idx %= self._main_len
            idx = self.main_idx_map[idx]
            return self.main_dataset.prepare_train_img_g(idx, 'main')
        elif self._mode == 'auxiliary':
            idx %= self._auxiliary_len
            idx = self.auxiliary_idx_map[idx]
            return self.auxiliary_dataset.prepare_train_img_g(idx, 'auxiliary')
        else:
            raise ValueError('not valid data type')

    def __len__(self) -> int:
        """Length of dataset."""
        if self._mode == 'main':
            return self._main_len
        elif self._mode == 'auxiliary':
            return self._auxiliary_len
        else:
            raise ValueError('not valid data type')

    def convert_main_to_auxiliary(self) -> None:
        """Convert main dataset to auxiliary dataset."""
        self._mode = 'auxiliary'
        self._set_group_flag()

    def save_data_infos(self, output_path: str) -> None:
        """Save data infos of main and auxiliary data."""
        self.main_dataset.save_data_infos(output_path)
        paths = output_path.split('.')
        self.auxiliary_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['auxiliary', paths[-1]]))

    def _set_group_flag(self) -> None:
        # disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @staticmethod
    def reweight_dataset(dataset: FewShotDiorDataset,
                         group_prefix: Sequence[str],
                         repeat_length: int = 100) -> List:
        """Reweight the dataset."""

        groups = [[] for _ in range(len(group_prefix))]
        for i in range(len(dataset)):
            filename = dataset.data_infos[i]['filename']
            for j, prefix in enumerate(group_prefix):
                if prefix in filename:
                    groups[j].append(i)
                    break
                assert j < len(group_prefix) - 1

        # Reweight the dataset to be consistent with the original
        # implementation of MPSR. For more details, please refer to
        # https://github.com/jiaxi-wu/MPSR/blob/master/maskrcnn_benchmark/data/datasets/voc.py#L137
        reweight_idx_map = []
        for g in groups:
            if len(g) < 50:
                reweight_idx_map += g * (int(repeat_length / len(g)) + 1)
            else:
                reweight_idx_map += g
        return reweight_idx_map
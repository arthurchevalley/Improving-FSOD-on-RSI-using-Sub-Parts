# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
import warnings
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Dict, Optional, Tuple

from mmdet.datasets import PIPELINES, DATASETS
#DATASETS = Registry('dataset')
#PIPELINES = Registry('pipeline')

from .samplers import (DistributedInfiniteGroupSampler, DistributedInfiniteSampler, InfiniteGroupSampler)
from mmdet.datasets.samplers import ClassAwareSampler, DistributedGroupSampler, DistributedSampler, GroupSampler, InfiniteBatchSampler, InfiniteGroupBatchSampler

from .two_branch_dataset import TwoBranchDataset
from .utils import get_copy_dataset_type
if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))



def _concat_dataset(cfg, default_args=None):
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                                   MultiImageMixDataset, RepeatDataset)

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)

    elif cfg['type'] == 'TwoBranchDataset':

        main_dataset = build_dataset(cfg['main_dataset'], default_args)
        # if `copy_from_main_dataset` is True, copy and update config
        # from main_dataset and copy `data_infos` by using copy dataset
        # to avoid reproducing random sampling.
        if cfg['auxiliary_dataset'].pop('copy_from_main_dataset', False):
            auxiliary_dataset_cfg = copy.deepcopy(cfg['main_dataset'])
            auxiliary_dataset_cfg.update(cfg['auxiliary_dataset'])
            auxiliary_dataset_cfg['type'] = get_copy_dataset_type(
                cfg['main_dataset']['type'])
            auxiliary_dataset_cfg['ann_cfg'] = [
                dict(data_infos=copy.deepcopy(main_dataset.data_infos))
            ]
            cfg['auxiliary_dataset'] = auxiliary_dataset_cfg
        auxiliary_dataset = build_dataset(cfg['auxiliary_dataset'],
                                          default_args)
        
        dataset = TwoBranchDataset(
            main_dataset=main_dataset,
            auxiliary_dataset=auxiliary_dataset,
            reweight_dataset=cfg.get('reweight_dataset', False))

    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     persistent_workers=False,
                     class_aware_sampler=None,
                     use_infinite_sampler = False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, Optional): Seed to be used. Default: None.
        runner_type (str): Type of runner. Default: `EpochBasedRunner`
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        class_aware_sampler (dict): Whether to use `ClassAwareSampler`
            during training. Default: None.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if runner_type == 'IterBasedRunner':
        # this is a batch sampler, which can yield
        # a mini-batch indices each time.
        # it can be used in both `DataParallel` and
        # `DistributedDataParallel`
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(
                dataset,
                batch_size,
                world_size,
                rank,
                seed=seed,
                shuffle=False)
        batch_size = 1
        sampler = None
    else:
        if class_aware_sampler is not None:
            # ClassAwareSampler can be used in both distributed and
            # non-distributed training.
            num_sample_class = class_aware_sampler.get('num_sample_class', 1)
            sampler = ClassAwareSampler(
                dataset,
                samples_per_gpu,
                world_size,
                rank,
                seed=seed,
                num_sample_class=num_sample_class)
        elif dist:
            # DistributedGroupSampler will definitely shuffle the data to
            # satisfy that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset,
                                   samples_per_gpu) if shuffle else None
        batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (TORCH_VERSION != 'parrots'
            and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')
    if isinstance(dataset, TwoBranchDataset):
        from ..utils import multi_pipeline_collate_fn
        from .dataloader_wrappers import TwoBranchDataloader

        # `TwoBranchDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]
        # initialize main dataloader
        main_data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # convert main dataset to auxiliary dataset
        auxiliary_dataset = copy.deepcopy(dataset)
        auxiliary_dataset.convert_main_to_auxiliary()
        # initialize auxiliary sampler and dataloader
        auxiliary_samples_per_gpu = samples_per_gpu
        auxiliary_workers_per_gpu =  workers_per_gpu
        (auxiliary_sampler, auxiliary_batch_size,
         auxiliary_num_workers) = build_sampler(
             dist=dist,
             shuffle=shuffle,
             dataset=auxiliary_dataset,
             num_gpus=num_gpus,
             samples_per_gpu=auxiliary_samples_per_gpu,
             workers_per_gpu=auxiliary_workers_per_gpu,
             seed=seed,
             use_infinite_sampler=use_infinite_sampler)
        auxiliary_data_loader = DataLoader(
            auxiliary_dataset,
            batch_size=auxiliary_batch_size,
            sampler=auxiliary_sampler,
            num_workers=auxiliary_num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn,
                samples_per_gpu=auxiliary_samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # wrap two dataloaders with dataloader wrapper
        data_loader = TwoBranchDataloader(
            main_data_loader=main_data_loader,
            auxiliary_data_loader=auxiliary_data_loader)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=kwargs.pop('pin_memory', False),
            worker_init_fn=init_fn,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def build_sampler(
        dist: bool,
        shuffle: bool,
        dataset: Dataset,
        num_gpus: int,
        samples_per_gpu: int,
        workers_per_gpu: int,
        seed: int,
        use_infinite_sampler: bool = False) -> Tuple[Sampler, int, int]:
    """Build pytorch sampler for dataLoader.
    Args:
        dist (bool): Distributed training/test or not.
        shuffle (bool): Whether to shuffle the data at every epoch.
        dataset (Dataset): A PyTorch dataset.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        seed (int): Random seed.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.
    Returns:
        tuple: Contains corresponding sampler and arguments
            - sampler(:obj:`sampler`) : Corresponding sampler
              used in dataloader.
            - batch_size(int): Batch size of dataloader.
            - num_works(int): The number of processes loading data in the
                data loader.
    """

    rank, world_size = get_dist_info()
    if dist:
        # Infinite sampler will return a infinite stream of index. But,
        # the length of infinite sampler is set to the actual length of
        # dataset, thus the length of dataloader is still determined
        # by the dataset.

        if shuffle:
            if use_infinite_sampler:
                sampler = DistributedInfiniteGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                # DistributedGroupSampler will definitely shuffle the data to
                # satisfy that images on each GPU are in the same group
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            if use_infinite_sampler:
                sampler = DistributedInfiniteSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if use_infinite_sampler:
            sampler = InfiniteGroupSampler(
                dataset, samples_per_gpu, seed=seed, shuffle=shuffle)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) \
                if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    return sampler, batch_size, num_workers
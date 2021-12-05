import itertools
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner.dist_utils import get_dist_info
from mmdet.datasets.samplers import (DistributedGroupSampler,
                                     DistributedSampler, GroupSampler)
from tabulate import tabulate
from termcolor import colored
from torch.utils.data import DataLoader

from .infinite_sampler import InfiniteBatchSampler, InfiniteGroupBatchSampler


def print_instances_class_histogram(infos, class_names, total_img_nums=None):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    rank, _ = get_dist_info()
    num_classes = len(class_names)
    N_COLS = min(6, len(class_names) * 2)

    data = list(itertools.chain(*infos))
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(['total', total_num_instances])
    if total_img_nums is not None:
        data.extend(['total img', total_img_nums])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=['category', '#instances'] * (N_COLS // 2),
        tablefmt='pipe',
        numalign='left',
        stralign='center',
    )
    if rank == 0:
        print('Distribution of instances among all {} categories:\n'.format(
            num_classes) + colored(table, 'cyan'))


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
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
            batch_sampler = InfiniteGroupBatchSampler(dataset,
                                                      batch_size,
                                                      world_size,
                                                      rank,
                                                      seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(dataset,
                                                 batch_size,
                                                 world_size,
                                                 rank,
                                                 seed=seed,
                                                 shuffle=False)
        batch_size = 1
        sampler = None
    else:
        if dist:
            # DistributedGroupSampler will definitely shuffle the data to
            # satisfy that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(dataset,
                                                  samples_per_gpu,
                                                  world_size,
                                                  rank,
                                                  seed=seed)
            else:
                sampler = DistributedSampler(dataset,
                                             world_size,
                                             rank,
                                             shuffle=False,
                                             seed=seed)
        else:
            sampler = GroupSampler(dataset,
                                   samples_per_gpu) if shuffle else None
        batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             batch_sampler=batch_sampler,
                             collate_fn=partial(
                                 collate, samples_per_gpu=samples_per_gpu),
                             pin_memory=False,
                             worker_init_fn=init_fn,
                             **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

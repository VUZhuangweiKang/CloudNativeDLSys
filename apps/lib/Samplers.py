import math
import os
import json
from torch.utils.data import Dataset, Sampler
import numpy as np
from collections import defaultdict
import torch.distributed as dist
import itertools
from typing import (
    TypeVar,
    Iterator,
    Tuple,
    Optional,
    Sized,
    List,
    Iterable,
    Union
)

T_co = TypeVar('T_co', covariant=True)
NODE = os.environ.get('NODE_IP')


class CustomSequentialSampler(Sampler[Tuple[int]]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[Tuple[int]]:
        return iter(self.data_source)

    def __len__(self) -> int:
        return len(self.data_source)


def get_indices(dataset, num_workers, shuffle):
    block_indices = np.arange(len(dataset.data_files))
    np.random.shuffle(block_indices)
    block_idx_group = np.array_split(block_indices, num_workers)
    workers_indices = []
    for group in block_idx_group:
        indices = []
        for block_idx in group:
            blockinfo = dataset.data_files[block_idx]
            idxs_in_block = []
            for j in range(int(blockinfo[-1])):
                idxs_in_block.append((blockinfo[1], block_idx, j))
            if shuffle:
                np.random.shuffle(idxs_in_block)
            indices.append(idxs_in_block)
        if shuffle:
            np.random.shuffle(indices)
        indices = [tup for block in indices for tup in block]
        workers_indices.append(indices)
    return workers_indices


class CustomBatchSampler:
    def __init__(self, samplers: List[Union[Sampler[int], Iterable[int]]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        self.num_workers = len(samplers)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        worker_iters = itertools.cycle([(w, iter(sampler)) for w, sampler in enumerate(self.samplers)])
        worker_done = [0] * self.num_workers
        if self.drop_last:
            while True:
                try:
                    if sum(worker_done) == self.num_workers:
                        break
                    w, sampler_iter = next(worker_iters)
                    if worker_done[w]:
                        yield []
                        continue
                    
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    worker_done[w] = 1
                finally:
                    if sum(worker_done) == self.num_workers:
                        break
        else:
            while True:
                try:
                    if sum(worker_done) == self.num_workers:
                        break
                    batch = [0] * self.batch_size
                    idx_in_batch = 0
                    
                    w, sampler_iter = next(worker_iters)
                    if worker_done[w]:
                        yield []
                        continue
                    
                    for _ in range(self.batch_size):
                        batch[idx_in_batch] = next(sampler_iter)
                        idx_in_batch += 1
                    yield batch
                except StopIteration:
                    worker_done[w] = 1
                    if idx_in_batch > 0:
                        yield batch[:idx_in_batch]

    def __len__(self) -> int:
        total_samples = sum([len(self.samplers[i]) for i in range(len(self.samplers))])
        if self.drop_last:
            num_batches = total_samples // self.batch_size  # type: ignore[arg-type]
        else:
            num_batches = (total_samples + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
        return num_batches

def create_block_sampler(dataset, shuffle, num_workers, batch_size, drop_last):
    workers_indices = get_indices(dataset, num_workers, shuffle)
    samplers = []
    for indices in workers_indices:
        sampler = CustomSequentialSampler(indices)
        samplers.append(sampler)
    return CustomBatchSampler(samplers, batch_size, drop_last)


class CustomDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
    

    def assign_samples(self, samples_on_nodes, workers_on_nodes):
        # 初始化任务分配表
        samples_allocation = defaultdict(list)

        # 分配每个节点上的数据块
        block_indices = defaultdict(int)
        for node, workers in workers_on_nodes.items():
            for worker in workers:
                while block_indices[node] < len(samples_on_nodes[node]) and len(samples_allocation[worker]) < self.num_samples:
                    samples_allocation[worker].append(samples_on_nodes[node][block_indices[node]])
                    block_indices[node] += 1

        # 为剩余的 worker 分配数据块
        remaining_nodes = list(samples_on_nodes.keys())
        current_node = 0
        for node, workers in workers_on_nodes.items():
            for worker in workers:
                while len(samples_allocation[worker]) < self.num_samples:
                    while block_indices[remaining_nodes[current_node]] >= len(samples_on_nodes[remaining_nodes[current_node]]):
                        current_node = (current_node + 1) % len(remaining_nodes)
                    samples_allocation[worker].append(samples_on_nodes[remaining_nodes[current_node]][block_indices[remaining_nodes[current_node]]])
                    block_indices[remaining_nodes[current_node]] += 1

        return samples_allocation

    def get_indices(self):
        indices = []
        for i, blockinfo in enumerate(self.dataset.data_files):
            block = []
            for j in range(int(blockinfo[-1])):
                block.append((blockinfo[1], i, j))
            if self.shuffle:
                np.random.shuffle(block)
            indices.append(block)
        if self.shuffle:
            np.random.shuffle(indices)
        indices = [tup for block in indices for tup in block]
        return indices

    def __iter__(self) -> Iterator[T_co]:
        indices = self.get_indices()

        # Handle uneven dataset size by adding padding or removing tail
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        samples_on_nodes = defaultdict(list)
        for info in indices:
            samples_on_nodes[info[0]].append(info)
        
        rank = dist.get_rank()
        workers_on_nodes = defaultdict(list)
        # nodes in jobInfo must be sort by rank, otherwise rank != workers_on_nodes[node]
        with open('/meta/jobInfo.json') as f:
            meta = json.load(f)
            workers_nodes = meta['nodes']
        for i, node in enumerate(workers_nodes):
            workers_on_nodes[node].append(i)

        indices = self.assign_samples(samples_on_nodes, workers_on_nodes)[rank]
        
        assert len(indices) == self.num_samples        
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
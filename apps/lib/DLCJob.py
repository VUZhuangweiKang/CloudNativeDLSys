import os
import math
import numpy as np
import threading
import queue
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, Dataset, Sampler, _utils, BatchSampler
from torch.utils.data.dataloader import _BaseDataLoaderIter
from lib.Samplers import *
from collections import defaultdict
import time
import pickle
import zmq
import psutil
import abc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings
from typing import (
    Iterable,
    Callable,
    List,
    Any,
    Tuple,
    Dict,
    Union,
    Optional,
    Sequence,
    TypeVar,
)
import configparser


warnings.filterwarnings("ignore")

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]


def get_cpu_limit():
    with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
        cfs_quota_us = int(f.read())
    if cfs_quota_us == -1:
        return mp.cpu_count()
    with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
        cfs_period_us = int(f.read())
    cpu_count = cfs_quota_us / cfs_period_us
    return int(cpu_count)
cpu_count = get_cpu_limit()


class ConfigParser:
    def __init__(self, components: List[str]):
        self.configs: Dict[str, configparser.ConfigParser] = {}
        for component in components:
            config_file = f"/configs/{component}.conf"
            parser = configparser.ConfigParser()
            parser.read(config_file)
            self.configs[component] = parser

    def get(self, component: str):
        return self.configs[component]
    
parser = ConfigParser(components=['client', 'commons'])
zmq_configs = parser.get('client')['zmq']
common_configs = parser.get('commons')['data']
ssd_base_dir = common_configs['ssd_base_dir']
runtime_dir = '/runtime'
memory_watermark = 90

MERGE = int(os.getenv("MERGE"))
THREADPOOLSIZE = os.getenv("THREADPOOLSIZE")
THREADPOOLSIZE = 1 if len(THREADPOOLSIZE) == 0 else int(THREADPOOLSIZE)


class DLCJobDataset(Dataset[T_co], metaclass=abc.ABCMeta):
    def __init__(self, name: str):
        """An abstract class subclassing the torch.utils.data.Dataset class

        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`process`, supporting pre-processing loaded data.
        Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~DLCJobDataLoader`.

        .. note:: Subclassing ~DLCJobDataset will load data under provided keys from DLCache to var:`self._samples`
        as Map<Key, Value>. Overwriting meth:`process` allows you to replace var:`self._samples` and
        var:`self._targets` with iterable variables that can be iterated in meth:`__get_item__`.
        """

        self._name = name
        while not os.path.exists(f"/share/signal_from_client"):
            time.sleep(1)
        
        self.data_files = np.load(f"/share/{self._name}_blocks.npy", mmap_mode='r')
        # collect blocks for each node
        self.node_blocks_map = defaultdict(list)
        for i, info in enumerate(self.data_files):
            self.node_blocks_map[info[1]].append(i)

        self.num_samples = sum(self.data_files[:, -1].astype(int))
        self.num_blocks = len(self.data_files)
        self.cache_hits = 0
        self.io_times, self.processing_times = [], []
        
        self.block_data = defaultdict(list)
        self.block_executors = {}
        self.futures = defaultdict(list)
        self.num_unused_items = defaultdict(int)
        
        self.worker_idx = None

    @abc.abstractmethod
    def _process_item(self, item_cloud_path: str, contents: Any) -> Any:
        pass

    def getitem_from_block(self, index: Tuple):
        _, block_idx, item_idx = index
        chunk_cloud_path, loc, chunk_etag, block, _  = self.data_files[block_idx]

        if block_idx not in self.num_unused_items:
            # print(f"worker {self.worker_idx}: load new block {block_idx}")
            block_path = os.path.join(runtime_dir, loc, chunk_etag, block)
            if not os.path.exists(block_path):
                block_path = block_path.replace(runtime_dir, ssd_base_dir)
            
            # load the block & process items asynchronously
            s = time.time()
            with open(block_path, 'rb') as f:
                self.block_data[block_idx] = pickle.load(f)
            self.io_times.append(time.time() - s)
            
            self.num_unused_items[block_idx] = len(self.block_data[block_idx])
            self.block_executors[block_idx] = ThreadPoolExecutor(max_workers=THREADPOOLSIZE)
            for file_name, data in self.block_data[block_idx]:
                item_cloud_path=f"{chunk_cloud_path}/{file_name.split('/')[-1]}"
                future = self.block_executors[block_idx].submit(self._process_item, item_cloud_path, data)
                self.futures[block_idx].append(future)

        s = time.time()
        data = self.futures[block_idx][item_idx].result()
        self.num_unused_items[block_idx] -= 1
        
        # clear used block
        if self.num_unused_items[block_idx] == 0:
            # print(f"worker {self.worker_idx}: release block {block_idx}")
            self.block_executors[block_idx].shutdown()
            del self.block_executors[block_idx], self.futures[block_idx], self.num_unused_items[block_idx], self.block_data[block_idx]

        self.processing_times.append(time.time() - s)
        return data, None
    
    def default_getitem(self, index: Tuple):
        _, block_idx, item_idx = index
        chunk_cloud_path, loc, chunk_etag, block, _  = self.data_files[block_idx]
        block_path = os.path.join(runtime_dir, loc, chunk_etag, block)
        if not os.path.exists(block_path):
            block_path = block_path.replace(runtime_dir, ssd_base_dir)
    
        s = time.time()
        with open(block_path, 'rb') as f:
            data = f.read()
        self.io_times.append(time.time() - s)
        
        s = time.time()
        item_cloud_path=f"{chunk_cloud_path}/{block_path.split('/')[-1]}"
        data = self._process_item(item_cloud_path, data)
        self.processing_times.append(time.time() - s)

        return data, None

    def __getitem__(self, index) -> T_co:
        if MERGE:
            return self.getitem_from_block(index)
        else:
            return self.default_getitem(index)
    
    def __len__(self) -> int:
        return self.num_samples


class DLCJobDataLoader(DataLoader[T_co]):
    dataset: DLCJobDataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Sampler
    prefetch_factor: int
    _iterator: Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(self, dataset: DLCJobDataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 1, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                 persistent_workers: bool = False, autoscale_workers: bool = False, max_tune_iters: int = 20,
                 pin_memory_device: str = ""):

        if int(os.environ["WORLD_SIZE"]) > 1:
            sampler = CustomDistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
            super().__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=pin_memory, timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context, generator=generator,
                            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                            pin_memory_device=pin_memory_device)
        else: 
            batch_sampler = create_block_sampler(dataset, shuffle, num_workers, batch_size, drop_last)
            super().__init__(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=pin_memory, timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context, generator=generator,
                            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                            pin_memory_device=pin_memory_device)

        assert self.num_workers > 0
        assert self.prefetch_factor > 0

        self.__initialized = True
        self._iterator = None

        self.check_worker_number_rationality()
        self.num_batches = len(self)
        self.autoscale_workers = autoscale_workers
        self.tune_iters = 0
        self.max_tune_iters = max_tune_iters

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self._iterator is not None:
            self.tune_iters = self._iterator._tune_iters
            num_workers = self._iterator._active_workers.value
        elif self.autoscale_workers:
            num_workers = cpu_count
        else:
            num_workers = self.num_workers

        return _DLCJobDataLoaderIter(self, self.num_batches, num_workers, self.autoscale_workers, self.tune_iters,
                                     self.max_tune_iters)

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DLCJobDataLoader, self).__setattr__(attr, val)
        

class StatefulCycleIterator:
    def __init__(self, num_workers=0):
        self.init_num_workers = num_workers
        self._workers_status = [1 for _ in range(num_workers)]
        self._ptr = 0

    def __next__(self):
        for _ in range(len(self._workers_status)):
            if self._ptr >= len(self._workers_status):
                self._ptr = 0
            if self._workers_status[self._ptr] == 1:
                w = self._ptr
                self._ptr += 1
                return w
            else:
                self._ptr += 1
        return None

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._workers_status)

    def get_ptr(self):
        return self._ptr

    def set_ptr(self, pos):
        self._ptr = pos

    # append to the end
    def append(self, worker_id, status=1):
        assert worker_id == len(self._workers_status)
        self._workers_status.append(status)

    def set_status(self, index, status):
        assert index < len(self._workers_status)
        self._workers_status[index] = status

    def get_status(self, index):
        return self._workers_status[index]

    def reactive_worker(self):
        for i in range(len(self)):
            if self._workers_status[i] == 0:
                self._workers_status[i] = 1
                return True
        return False

    def deactive_worker(self):
        for i in range(len(self)):
            if self._workers_status[i] == 1:
                self._workers_status[i] = 0
                return True
        return False

    def reset(self):
        self._ptr = 0


class _DLCJobDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader, num_batches, opt_num_workers, autoscale_workers: bool = False, tune_iters: int = None,
                 max_tune_iters: int = None):
        super(_DLCJobDataLoaderIter, self).__init__(loader)

        self._num_batches = num_batches
        self._autoscale_workers = autoscale_workers
        self._num_workers = opt_num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._tune_freq = 1
        self._max_tune_iters = max_tune_iters

        # sockets for communication with client
        zmq_context = zmq.Context()
        self._socket_req = zmq_context.socket(zmq.REQ)
        self._socket_req.connect(zmq_configs['init_channel'])
        self._socket_pub = zmq_context.socket(zmq.PUB)
        self._socket_pub.bind(zmq_configs['ipc_channel'])

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context
        self._multiprocessing_context = multiprocessing_context
        self._worker_init_fn = loader.worker_init_fn

        # We don't consider DataPipe currently Adds forward compatibilities so classic DataLoader can work with
        # DataPipes: Additional worker init function will take care of sharding in MP and Distributed if isinstance(
        # self._dataset, (IterDataPipe, MapDataPipe)): self._worker_init_fn = functools.partial(
        # _sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank)

        self._active_workers = 0

        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False

        self._workers_done_event = []
        self._index_queues = []
        self._workers = []
        self._worker_queue_idx_cycle = StatefulCycleIterator()
        for _ in range(self._num_workers):
            self._spawn_worker()

        # send batch sampler indexes to client
        # inform client to init Cache
        sampler_iter = iter(self._index_sampler)
        batches = [batch for batch in sampler_iter]
        init_info = {'num_workers': self._num_workers, 'num_cores': cpu_count, 'batches': batches}
        self._socket_req.send_multipart([b"init", self._dataset._name.encode('utf-8'), pickle.dumps(init_info)])
        self._socket_req.recv()

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
            else:
                current_device = torch.cuda.current_device()  # choose cuda for default
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      current_device,
                      self._pin_memory_thread_done_event, self._pin_memory_device))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_DLCJobDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True

        self._tune_iters = tune_iters
        self._worker_num_hist = []
        self._memory_usage = []
        self._reset(loader, first_iter=True)

        # the belows are for benefit-aware dataset placement test
        """
        _data_load_time: duration of executing the _next_data function in DataLoader
        _computing_time: interval between consecutive call of the __next__ function in DataLoader
        """
        self._data_load_time = []
        self._computing_time = []

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)

        self._last_iter_time = None
        """
        # the below records will be reset after 1 worker tuning window
        # window_size = self._active_workers * self._prefetch_factor
        _req_time: interval between consecutive call of the __next__ function in DataLoader
        _fetch_time: time used by Worker to fetch 1 batch data
        """
        self._req_time = []
        self._fetch_time = []

        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__

        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)

        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._reorder_dict = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in _reorder_dict.values() if len(v) == 1)
        self._outstanding_idx_dict = defaultdict(int)

        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._active_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration(self._shared_seed))
            resume_iteration_cnt = self._active_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1

        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._active_workers):
            self._try_put_index()

        req_time, fetch_time = np.mean(self._req_time), np.mean(self._fetch_time)
        if req_time < fetch_time:
            msg = {'send_idx': self._prefetch_factor * self._active_workers,
                   'rcvd_idx': self._rcvd_idx,
                   'active_workers': self._active_workers}
            # self._socket_pub.send_multipart([b"loadCache", self._dataset._name.encode('utf-8'), pickle.dumps(msg)])

    def _spawn_worker(self):
        if self._worker_queue_idx_cycle is not None and self._worker_queue_idx_cycle.reactive_worker():
            self._active_workers += 1
            return

        worker_id = len(self._worker_queue_idx_cycle)
        worker_done_event = self._multiprocessing_context.Event()
        idx_queue = self._multiprocessing_context.Queue()
        idx_queue.cancel_join_thread()
        w = self._multiprocessing_context.Process(target=_utils.worker._worker_loop,
                                                  args=(self._dataset_kind, self._dataset, idx_queue,
                                                        self._worker_result_queue, worker_done_event,
                                                        self._auto_collation, self._collate_fn, self._drop_last,
                                                        self._base_seed, self._worker_init_fn,
                                                        worker_id, self._active_workers, self._persistent_workers))

        w.daemon = True
        w.start()
        self._workers_done_event.append(worker_done_event)
        self._index_queues.append(idx_queue)
        self._workers.append(w)
        if self._worker_queue_idx_cycle is not None:
            self._worker_queue_idx_cycle.append(worker_id)
        self._active_workers += 1

    def _pause_worker(self):
        if self._worker_queue_idx_cycle.deactive_worker():
            self._active_workers -= 1

    def _tune_worker_num(self):
        if self._tune_iters >= self._max_tune_iters:
            if len(self._outstanding_idx_dict) > self._active_workers:
                for i, worker_status in enumerate(self._worker_queue_idx_cycle._workers_status):
                    if self._outstanding_idx_dict[i] == 0 and worker_status == 0 and self._index_queues[i].qsize() == 0:
                        self._workers_done_event[i].set()
                        self._mark_worker_as_unavailable(i, shutdown=True)
                        self._workers[i].join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                        self._index_queues[i].cancel_join_thread()
                        self._index_queues[i].close()
                        self._workers[i].terminate()
                        del self._outstanding_idx_dict[i]
                return
            else:
                return

        if self._rcvd_idx % self._tune_freq != 0:
            return

        est_num_workers = self._active_workers
        # estimate required workers by comparing req and load time
        if len(self._fetch_time) > 0 and len(self._req_time) > 0:
            est_num_workers = math.ceil(np.mean(self._fetch_time) / np.mean(self._req_time))

        if est_num_workers > cpu_count:
            mem_usage = psutil.virtual_memory().percent
            # try to set the num_workers = cpu_cores, but avoid overflow memory
            avg_memory_usage_per_worker = mem_usage / self._active_workers
            if avg_memory_usage_per_worker * est_num_workers > memory_watermark:
                est_num_workers = math.floor(memory_watermark / avg_memory_usage_per_worker)

        new_num_workers = min(est_num_workers, cpu_count)

        # print('current worker num: {}, estimated worker num: {}, new worker num: {}'.format(num_workers,
        # est_num_workers, new_num_workers)) commit the tunning action
        delta = new_num_workers - self._active_workers
        for _ in range(abs(delta)):
            if delta > 0:
                self._spawn_worker()
            elif delta < 0:
                self._pause_worker()

        if delta > 0:
            for _ in range(delta):
                self._try_put_index()

        # print('change worker num to {}'.format(new_num_workers))
        self._worker_num_hist.append(new_num_workers)
        self._tune_iters += 1

    def _try_put_index(self):
        while True:
            try:
                batched_idxs = self._next_index()
            except StopIteration:
                return

            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if worker_queue_idx is None:
                return
            if len(batched_idxs) == 0:
                continue
            else:
                break

        self._index_queues[worker_queue_idx].put((self._send_idx, batched_idxs))
        self._reorder_dict[self._send_idx] = (worker_queue_idx,)
        self._outstanding_idx_dict[worker_queue_idx] += 1
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return True, data
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._worker_queue_idx_cycle.get_status(worker_id) != -1 and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError(f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly") from e
            if isinstance(e, queue.Empty):
                return False, None
            import tempfile
            import errno
            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise

    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _process_data(self, data):
        self._rcvd_idx += 1
        if self._send_idx < self._rcvd_idx + self._active_workers * self._prefetch_factor:
            self._try_put_index()
        if isinstance(data, _utils.worker.ExceptionWrapper):
            data.reraise()
        return data

    def _next_data(self):
        start = time.time()
        if self._last_iter_time is not None and self._tune_iters < self._max_tune_iters:
            self._computing_time.append(start-self._last_iter_time)
            window_size = self._active_workers * self._prefetch_factor
            if len(self._req_time) == window_size:
                self._req_time.clear()
            if len(self._fetch_time) == window_size:
                self._fetch_time.clear()
            self._req_time.append(start - self._last_iter_time)

        req_time_, fetch_time_ = np.mean(self._req_time), np.mean(self._fetch_time)
        if req_time_ < fetch_time_:
            msg = {'send_idx': self._send_idx + 1,
                   'rcvd_idx': self._rcvd_idx,
                   'active_workers': self._active_workers}
            # self._socket_pub.send_multipart([b"loadCache", self._dataset._name.encode('utf-8'), pickle.dumps(msg)])
        
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._reorder_dict[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._worker_queue_idx_cycle.get_status(worker_id) != -1:  # has data or is still active
                    break
                del self._reorder_dict[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                if not self._persistent_workers:
                    # self._socket_pub.send_multipart([b"expireChunk", self._dataset._name.encode('utf-8'), b""])
                    self._shutdown_workers()

                # epoch is down
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch
            # Check if the next sample has already been generated
            if len(self._reorder_dict[self._rcvd_idx]) == 2:
                data, worker_id, fetch_time = self._reorder_dict.pop(self._rcvd_idx)[1]
                data = self._process_data(data)
                if self._tune_iters < self._max_tune_iters and fetch_time is not None:
                    self._fetch_time.append(fetch_time) 
                break

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data, worker_id, fetch_time, miss = self._get_data()
            if self._tune_iters < self._max_tune_iters and fetch_time is not None:
                self._fetch_time.append(fetch_time)

            if miss:
                self._socket_pub.send_multipart([b'missETags', self._dataset._name.encode('utf-8'), pickle.dumps(miss)])

            self._tasks_outstanding -= 1
            self._outstanding_idx_dict[worker_id] -= 1

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._reorder_dict[idx] += ((data, worker_id, fetch_time),)
            else:
                del self._reorder_dict[idx]
                data = self._process_data(data)
                break

        if self._autoscale_workers:
            self._tune_worker_num()

        mem_usage = psutil.virtual_memory().percent
        self._memory_usage.append(mem_usage)

        # self._socket_pub.send_multipart([b"releaseCache", self._dataset._name.encode('utf-8'), str(self._rcvd_idx - 1).encode('utf-8')])

        self._last_iter_time = time.time()
        load_data_time = self._last_iter_time - start
        self._data_load_time.append(load_data_time)
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._worker_queue_idx_cycle.get_status(worker_id) != -1 or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._worker_queue_idx_cycle.set_status(worker_id, -1)

        assert self._workers_done_event[worker_id].is_set() == shutdown

    def _shutdown_workers(self):
        np.save('/share/data_load_time.npy', self._data_load_time)
        np.save('/share/compute_time.npy', self._computing_time)
        if self._autoscale_workers:
            np.save('/share/worker_num_hist.npy', self._worker_num_hist)
            np.save('/share/memory_usage.npy', self._memory_usage)

        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                for worker_id in range(len(self._workers)):
                    if not self._workers_done_event[worker_id].is_set():
                        self._workers_done_event[worker_id].set()
                        # Get number of workers from `len(self._workers)` instead of
                        # `self._num_workers` in case we error before starting all
                        # workers.
                        # If we are using workers_status with persistent_workers
                        # we have to shut it down because the worker is paused
                        if self._persistent_workers or self._worker_queue_idx_cycle.get_status(worker_id) != -1:
                            self._mark_worker_as_unavailable(worker_id, shutdown=True)

                        # We should be able to join here, but in case anything went
                        # wrong, we set a timeout and if the workers fail to join,
                        # they are killed in the `finally` block.
                        self._workers[worker_id].join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                        self._index_queues[worker_id].cancel_join_thread()
                        self._index_queues[worker_id].close()
            finally:
                # close data queue
                while True:
                    running_workers = len(self._workers)
                    for done_event in self._workers_done_event:
                        if done_event.is_set():
                            running_workers -= 1
                    if running_workers == 0:
                        self._data_queue.cancel_join_thread()
                        self._data_queue.close()
                        break

                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()

        self._socket_pub.send_multipart([b'stopIteration', self._dataset._name.encode('utf-8'), b''])

    # staticmethod is used to remove reference to `_MultiProcessingDataLoaderIter`
    @staticmethod
    def _clean_up_worker(w):
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self):
        self._shutdown_workers()
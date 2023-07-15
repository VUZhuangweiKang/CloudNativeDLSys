import time
import shutil
import zmq
import pickle
import socket
import signal
import glob
import bson
import numpy as np
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
import threading
from typing import List, Dict, Union, Tuple
import databus.dbus_pb2 as pb
from database import DLCacheCollection, MongoOperator, ChunkStatus
from configurations import ConfigParser
import multiprocessing
from utils import *
from logger import get_logger


logger = get_logger(__name__, level='Debug')
parser = ConfigParser(components=['commons', 'client', 'dbus'])
dbus_configs, client_configs = parser.get('dbus')['manager'], parser.get('client')
init_channel = client_configs['zmq']['init_channel']
ipc_channel = client_configs['zmq']['ipc_channel']
ssd_base_dir = parser.get('commons')['data']['ssd_base_dir']
runtime_base_dir = '/runtime'
cooldown_sec = int(client_configs['data']['cooldown_sec'])

context = zmq.Context()
manager_uri = f"{dbus_configs['hostname']}:{dbus_configs['port']}"
mngr_socket = context.socket(zmq.REQ)
mngr_socket.connect(f"tcp://{manager_uri}")

PROBE = int(os.getenv("PROBE"))
MERGE = int(os.getenv("MERGE"))


def clear_runtime():
    for root, dirs, files in os.walk(runtime_base_dir, topdown=False):
        while True:
            try:
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    shutil.rmtree(os.path.join(root, name))
                break
            except:
                pass


class Client(object):
    worker_id = socket.gethostname()
    if PROBE:
        worker_id = worker_id.split('.')[0]
    worker_idx = int(worker_id[-1])
    job_id = '-'.join(worker_id.split('-')[:-1])
    
    def __init__(self):
        self.mongo_operator = MongoOperator.AsyncMongoOperator()
        
        # waiting for data preparation
        while True:
            pipeline = [
                {"$addFields": {"MetaId": {"$concat": ["$Meta.DLTDeployId", "-", "$Meta.JobId"]}}},
                {"$match": {"MetaId": self.job_id}}
            ]
            result_cursor = self.mongo_operator.job_col.aggregate(pipeline)
            if result_cursor.alive:
                break
            time.sleep(1)
        
        # for exp only
        # os.system("vmtouch -e /mnt/nfs/ssd")
        
        self.create_mappings()

        # DL application is blocked until this file is writen
        open(f"/share/signal_from_client", 'w').close()
        logger.info(f"Data preparation for job {self.job_id} done.")

        self.tmpfs_paths: Dict[str, List[List[str]]] = defaultdict(list)
        self.load_cache_proc = None
        self.release_cache_proc = None
        self.cool_down_proc = None

        self.send_idx_queue = None
        self.rcvd_idx_queue = None

        self.cache_usage = []
        self.mp_manager = None

        self.block_idx = None

        if self.worker_idx == 0:
            self.mongo_operator.update_job(self.job_id, {"Status": "running"})
            logger.info(f"Job {self.job_id} is running.")
        self.process_events()

    # create object mappings between cloud objects and NFS files
    def create_mappings(self):
        job = self.mongo_operator.find_job(self.job_id)
        job_info = {"ChunkETags": job["ChunkETags"], "Meta": job["Meta"]}
        assert job_info

        def process_chunk(chunk):
            cloud_path, loc, chunk_etag = chunk['Key'], chunk["Location"], chunk['ETag']
            ssd_path = f"{ssd_base_dir}/{loc}/{chunk_etag}"            
            count_blocks = defaultdict(int)
            
            if MERGE:
                blocks = ['/'.join(p.split('/')[-2:]) for p in glob.glob(f"{ssd_path}/merged*")]
                for block_name in blocks:
                    for block in chunk['Blocks']:
                        if block_name in block['Name']:
                            count_blocks[block_name] = block['Length']
                            break
            else:
                while True:
                    blocks = [f"{chunk_etag}/{p}" for p in os.listdir(ssd_path)]
                    if len(blocks) > 0:
                        break
                for block_name in blocks:
                    if 'merged' not in block_name:
                        count_blocks[block_name] = 1
            logger.info(f"Num blocks under chunk {chunk['ETag']}: {len(count_blocks)}")
            for block in blocks:
                yield [cloud_path, loc, chunk_etag, block.split('/')[-1], count_blocks[block]]
        
        # this will save all chunks info, including the chunks that might won't be used by this worker
        for dataset_name in job_info['ChunkETags']:
            etags = job_info['ChunkETags'][dataset_name]
            logger.info(f"{dataset_name} ETags: {etags}")
            if etags:
                chunks_iter = self.mongo_operator.project_etags(etags)
                manifest_gen = []
                for chunk in chunks_iter:
                    if MERGE and 'Blocks' not in chunk:
                        continue
                    ssd_path = f"{ssd_base_dir}/{chunk['Location']}/{chunk['ETag']}"
                    while True:
                        if os.path.exists(ssd_path):
                            for entry in process_chunk(chunk):
                                manifest_gen.append(entry)
                            break
                logger.info(f'Toal blocks for {dataset_name}: {len(manifest_gen)}')
                np.save(f"/share/{dataset_name}_blocks.npy", manifest_gen)

    def load_cache(self, send_idx_queue: mp.Queue, tmpfs_paths:  Dict[str, List[List[str]]], current_block):
        event = mp.Event()
        def copy_file(files):
            for tmpfs_path in files:
                if not event.is_set():
                    break
                nfs_path = tmpfs_path.replace(runtime_base_dir, ssd_base_dir)
                try:
                    while True:
                        try:
                            shutil.copyfile(nfs_path, tmpfs_path)
                            break
                        except FileNotFoundError:
                            root_folder = '/'.join(tmpfs_path.split('/')[:-1])
                            os.makedirs(root_folder, exist_ok=True)
                except FileNotFoundError as err:
                    logger.error(f"failed to copy {err.filename}")
                    etag = nfs_path.split('/')[-1]
                    
                    req = pb.DataMissRequest()
                    req.etag = etag
                    mngr_socket.send_multipart([b"handle_datamiss", req.SerializeToString()])
                    mngr_socket.recv()

        copy_file_thread = None
        while True:
            dataset_name, send_idx = send_idx_queue.get()
            num_batches = len(tmpfs_paths[dataset_name])
            if event.is_set():
                event.clear()

            if send_idx < num_batches:
                items = set()
                for block_path in tmpfs_paths[dataset_name][send_idx][::-1]:
                    # The shared current_block value can indicate whether to load/release the block
                    if current_block.value != block_path:
                        current_block.value = block_path
                        items.add(block_path)
                while copy_file_thread is not None and copy_file_thread.is_alive():
                    pass
                else:
                    event.set()
                    copy_file_thread = threading.Thread(target=copy_file, args=(list(items),), daemon=True)
                    copy_file_thread.start()

                # update chunk status
                chunk_etags = []
                for block_path in tmpfs_paths[dataset_name][send_idx]:
                    etag = block_path.split('/')[-2]
                    chunk_etags.append(etag)
                chunk_etags = list(set(chunk_etags))
                now = datetime.utcnow().timestamp()
                self.mongo_operator.execute(col=DLCacheCollection.Dataset,
                                            func="update_many",
                                            args=[{"ETag": {"$in": chunk_etags}},
                                                  {"$set": {"Status.code": ChunkStatus.ACTIVE},
                                                   "$inc": {"Status.active_count": 1},
                                                   "$push": {"References": bson.timestamp.Timestamp(int(now), inc=1)}}])
                del chunk_etags

    def release_cache(self, rcvd_idx_queue: mp.Queue, tmpfs_paths: Dict[str, List[List[str]]], current_bock):
        while True:
            dataset_type, rcvd_idx = rcvd_idx_queue.get()
            self.cache_usage.append(count_files(runtime_base_dir))
            if rcvd_idx < len(tmpfs_paths[dataset_type]):
                for block_path in tmpfs_paths[dataset_type][rcvd_idx][::-1]:
                    # delete the block only if the block path have changed
                    if len(current_bock.value) > 0 and current_bock.value != block_path:
                        try:
                            os.remove(block_path)
                            # logger.info(f"release cache {block_path}")
                        except FileNotFoundError:
                            pass

    def expire_chunks(self, dataset_name: str, tmpfs_paths: Dict[str, List[List[str]]]):
        time.sleep(cooldown_sec)
        etags = set()
        for batch in tmpfs_paths[dataset_name]:
            for block_path in batch:
                etags.add(block_path.split('/')[-2])
        etags = list(etags)
        now = datetime.utcnow().timestamp()
        self.mongo_operator.execute(col=DLCacheCollection.Dataset,
                                    func="update_many",
                                    args=[{"ETag": {"$in": etags}, "Status.active_count": 1},
                                          {"$set": {"Status.code": ChunkStatus.INACTIVE,
                                                    "Status.active_count": 0,
                                                    "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)}
                                           }])

        self.mongo_operator.execute(col=DLCacheCollection.Dataset,
                                    func="update_many",
                                    args=[{"ETag": {"$in": etags}, "Status.active_count": {"$gt": 1}},
                                          {"$inc": {"Status.active_count": -1},
                                           "$set": {"Status.code": ChunkStatus.INACTIVE,
                                                    "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)}
                                           }])

    def check_data_stall(self, data_loading_time: List) -> bool:
        # Normalize the data
        data_loading_time = (data_loading_time - np.mean(data_loading_time)) / np.std(data_loading_time)

        # Compute autocorrelation
        autocorr = np.correlate(data_loading_time, data_loading_time, mode='full')

        # Only consider the second half and ignore the first peak at lag zero
        autocorr = autocorr[len(autocorr)//2 + 1:]

        # Find lag of the peak
        lag_of_peak = np.argmax(autocorr) + 1

        # We can consider that there is a data stall if the peak is significantly higher than the average
        return autocorr[lag_of_peak - 1] > np.mean(autocorr) + 2 * np.std(autocorr)


    def save_perf_results(self, num_dataloading_workers, num_cores, batch_size):
        """
        data_load_time: duration of executing the _next_data function in DataLoader
        computing_time: interval between consecutive call of the __next__ function in DataLoader
        io_time: time used function to load 1 sample (I/O time)
        processing_time: time used function to process 1 sample
        """
        computing_time = np.load('/share/compute_time.npy', allow_pickle=True)
        data_load_time = np.load('/share/data_load_time.npy', allow_pickle=True)
        has_data_stall = self.check_data_stall(data_load_time)
        
        io_time = []
        for path in glob.glob('/share/io_time_*.npy'):
            io_time.extend(np.load(path, allow_pickle=True))
        processing_time = []
        for path in glob.glob('/share/processing_time_*.npy'):
            processing_time.extend(np.load(path, allow_pickle=True))
        self.mongo_operator.update_job(self.job_id, {
            "Performance": {
                "worker": self.worker_id,
                "values": {
                    "NumDataLoadingWorkers": num_dataloading_workers,
                    "NumCores": num_cores,
                    "BatchSize": batch_size,
                    "HasDataStall": bool(has_data_stall),
                    "ComputingTime": np.mean(computing_time),
                    "DataLoadingTime": np.mean(data_load_time),
                    "IOTime": np.mean(io_time) * batch_size,
                    "ProcessingTime": np.mean(processing_time) * batch_size
                }
            }
        }, operation='$push')
        ########## the belows are for experiment
        if not PROBE:
            perf_dir = f"/mnt/nfs/hdd/{os.getenv('NODE_IP')}/{socket.gethostname()}"
            if not os.path.exists(perf_dir):
                os.makedirs(perf_dir)
            np.save(f"{perf_dir}/compute_time.npy", computing_time)
            np.save(f"{perf_dir}/data_load_time.npy", data_load_time)
            np.save(f"{perf_dir}/io_time.npy", io_time)
            np.save(f"{perf_dir}/processing_time.npy", processing_time)
    
    def process_events(self):
        clear_runtime()
        context = zmq.Context()
        socket_rep = context.socket(zmq.REP)
        socket_rep.bind(init_channel)
        socket_sub = context.socket(zmq.SUB)
        socket_sub.connect(ipc_channel)
        for topic in [b'loadCache', b'releaseCache', b'expireChunk', b'stopIteration', b'missETags']:
            socket_sub.setsockopt(zmq.SUBSCRIBE, topic)

        poller = zmq.Poller()
        poller.register(socket_rep, zmq.POLLIN)
        poller.register(socket_sub, zmq.POLLIN)
        
        self.load_cache_processes: Dict[str, mp.Process] = {}
        self.release_cache_processes: Dict[str, mp.Process] = {}
        batch_size = 1
        num_workers = None
        num_cores = None
        
        while True:
            socks = dict(poller.poll())
            if socket_rep in socks and socks[socket_rep] == zmq.POLLIN:
                topic, dataset_name, data = socket_rep.recv_multipart()
                topic, dataset_name = topic.decode("utf-8"), dataset_name.decode('utf-8')
                logger.info(f"Process event: {topic} for dataset {dataset_name}")
                if topic == "init":
                    data = pickle.loads(data)
                    num_workers = data['num_workers']
                    num_cores = data['num_cores']
                    batches_idx: List[List[Tuple[int]]] = data['batches']
                    blocks_info = np.load(f"/share/{dataset_name}_blocks.npy")
                    # this is batched item path, every item is in the format: [block_path, file_name]
                    batched_tmpfs_paths: List[List] = []
                    for batch_idx in batches_idx:
                        batch = set()
                        for _, block_idx, _ in batch_idx:
                            block_path = os.path.join(*blocks_info[block_idx][1:-1].tolist())
                            batch.add(block_path)
                        batch = list(batch)
                        for i in range(len(batch)):
                            batch[i] = batch[i].replace(ssd_base_dir, runtime_base_dir)
                        batched_tmpfs_paths.append(batch)
                    self.tmpfs_paths[dataset_name] = batched_tmpfs_paths
                    batch_size = len(batched_tmpfs_paths[0])

                    self.send_idx_queue = mp.Queue()
                    self.send_idx_queue.cancel_join_thread()
                    self.rcvd_idx_queue = mp.Queue()
                    self.rcvd_idx_queue.cancel_join_thread()

                    self.mp_manager = mp.Manager()
                    self.current_block = self.mp_manager.Value(str, "")

                    self.load_cache_processes[dataset_name] = mp.Process(target=self.load_cache, args=(self.send_idx_queue, self.tmpfs_paths, self.current_block), daemon=True)
                    self.release_cache_processes[dataset_name] = mp.Process(target=self.release_cache, args=(self.rcvd_idx_queue, self.tmpfs_paths, self.current_block), daemon=True)
                    self.load_cache_processes[dataset_name].start()
                    self.release_cache_processes[dataset_name].start()
                    if not self.mongo_operator.started:
                        self.mongo_operator.start()
                    socket_rep.send(b'')
                    del batched_tmpfs_paths
            elif socket_sub in socks and socks[socket_sub] == zmq.POLLIN:
                topic, dataset_name, data = socket_sub.recv_multipart()
                topic, dataset_name = topic.decode("utf-8"), dataset_name.decode('utf-8')
                # logger.info(f"Process event: {topic} for dataset {dataset_name}")
                # logger.info('recv msg: {} {}'.format(topic, data))
                if topic == "loadCache":
                    data = pickle.loads(data)
                    if data['rcvd_idx'] == len(self.tmpfs_paths[dataset_name]):
                        continue
                    self.send_idx_queue.put((dataset_name, data['send_idx']))
                elif topic == "releaseCache":
                    idx = int(data)
                    self.rcvd_idx_queue.put((dataset_name, idx))
                elif topic == "expireChunk":
                    if self.cool_down_proc is not None and self.cool_down_proc.is_alive():
                        self.cool_down_proc.terminate()
                    self.cool_down_proc = mp.Process(target=self.expire_chunks, args=(dataset_name, self.tmpfs_paths), daemon=True)
                    self.cool_down_proc.start()
                elif topic == "stopIteration":
                    if len(self.cache_usage) > 0:
                        np.save(f"/share/{dataset_name}_cache_usage.npy", self.cache_usage)
                    clear_runtime()
                    if dataset_name in self.load_cache_processes and self.load_cache_processes[dataset_name].is_alive():
                        self.load_cache_processes[dataset_name].terminate()
                        del self.load_cache_processes[dataset_name]
                    if dataset_name in self.release_cache_processes and self.release_cache_processes[dataset_name].is_alive():
                        self.release_cache_processes[dataset_name].terminate()
                        del self.release_cache_processes[dataset_name]
                    if len(self.load_cache_processes) == 0 and len(self.release_cache_processes) == 0:
                        assert num_workers is not None
                        self.save_perf_results(num_workers, num_cores, batch_size)
                        
                    # for exp only
                    req = pb.DelDLTPodRequest()
                    req.dltpod = self.worker_id
                    mngr_socket.send_multipart([b"del_dltpod", req.SerializeToString()])
                    mngr_socket.recv()
                    self.mongo_operator.terminate()
                    self.mp_manager.shutdown()
                elif topic == "missETags":
                    miss_etags = pickle.loads(data)
                    for etag in miss_etags:
                        req = pb.DataMissRequest()
                        req.etag = etag
                        mngr_socket.send_multipart([b"handle_datamiss", req.SerializeToString()])
                        mngr_socket.recv()

                del socks, topic, dataset_name, data


if __name__ == '__main__':
    client = Client()
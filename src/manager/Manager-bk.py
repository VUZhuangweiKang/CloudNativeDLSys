import os.path
import time
import multiprocessing as mp
import concurrent.futures
import numpy as np
import pickle as pkl
from typing import List, Dict, Union, Tuple
from scipy.optimize import minimize_scalar
import zmq
import getpass
import random
import glob
import requests
import psutil
import json
import faulthandler
import pandas as pd

import sys
sys.path.append("../")
from manager import *
from manager.Placement import *
from manager.ChunkOps import *

faulthandler.enable()


def eval_resource_gap(workers_url, jobs: List[JobQueueItem], download=True) -> Tuple[float]:
    gang = len(jobs) > 1
    job = jobs[0]
    distributed = job.spec['numWorkers'] > 1
    cred, bucket = job.cred, job.spec['datasource']['bucket']

    # Check if HDDs have enough space for the ssd chunks. Compressed chunks are downloaded on HDD layer,
    # We employ Cost-Aware LRFU for both HDD and SSD layers.
    free_gpus = k8s_operator.get_free_gpus()
    # we don't want to chang the actual free_gpus record because here we only decide where to place the jobs
    # the actuall placement operation is conducted by the executor.
    free_gpus_bk = free_gpus
    free_hdd_space = k8s_operator.get_free_storage(base_dir=hdd_dir)
    preemptive_chunks_on_hdd = cost_aware_lrfu(ssd=False)
    
    if gang:
        # hdd_gap = multi_job_placement(free_gpus, free_hdd_space, preemptive_chunks_on_hdd, jobs, False, workers_url)
        hdd_gap = multi_job_placement(free_gpus, free_hdd_space, preemptive_chunks_on_hdd, jobs, ssd=False)
    elif distributed:
        hdd_gap = multi_worker_placement(free_gpus, free_hdd_space, preemptive_chunks_on_hdd, job, ssd=False)
    else:
        hdd_gap = base_placement(free_gpus, free_hdd_space, preemptive_chunks_on_hdd, job, ssd=False)
        
    logger.info(f"HDD storage gap: {hdd_gap}")
    
    # can't meet computation resource requirement or some ssd chunks can't be downloaded
    if hdd_gap is None:
        return np.inf, np.inf
    elif hdd_gap > 0:
        return hdd_gap, np.inf

    if download:
        context = zmq.Context()
        futures = []
        s3_operator = S3Operator(mongo_operator.get_s3auth(**cred))
        chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for chunk in jobs[0].chunks:
                if not chunk['ExistOnHDD']:
                    futures.append(executor.submit(download_chunk, workers_url[chunk['SourceLocation']], chunk, s3_operator, bucket, context))
                else:
                    logger.info(f"skip chunk {chunk['ETag']}")
                    chunks.append(chunk)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                chunks.append(result)
        for i in range(len(jobs)):
            jobs[i].chunks = chunks

    # Now, we have downloaded ssd chunks and know the actual chunk size. So the returned gap value is
    # based on the extracted chunk size. This step will also update the chunk location if the node is unable to
    # store the extracted chunks.
    free_gpus = free_gpus_bk
    free_ssd_space = k8s_operator.get_free_storage(base_dir=ssd_dir)
    preemptive_chunks_on_ssd = cost_aware_lrfu(ssd=True)
    if gang:
        ssd_gap = multi_job_placement(free_gpus, free_ssd_space, preemptive_chunks_on_ssd, jobs, ssd=True)
    elif distributed:
        ssd_gap = multi_worker_placement(free_gpus, free_ssd_space, preemptive_chunks_on_ssd, job, ssd=True)
    else:
        ssd_gap = base_placement(free_gpus, free_ssd_space, preemptive_chunks_on_ssd, job, ssd=True)
    
    # save nodes information into database
    for job in jobs:
        mongo_operator.update_job(f"{job.dltdeploy_id}-{job.job_id}", update={"Nodes": job.nodes})

    logger.info(f"SSD storage gap: {ssd_gap}")
    
    return hdd_gap, ssd_gap


def get_target_ssd():
    def is_ssd(device):
        """Check if the given device is an SSD."""
        try:
            with open(f'/sys/block/{os.path.basename(device)}/queue/rotational', 'r') as f:
                return f.read() == '0\n'
        except FileNotFoundError:
            return False
        
    # 获取所有磁盘分区信息
    partitions = psutil.disk_partitions()

    # 使用一个字典保存每个SSD设备名称和对应的总存储空间
    ssd_devices = {device.device: psutil.disk_usage(device.mountpoint).total for device in partitions if is_ssd(device.device)}

    # 找到存储空间最大的SSD设备名称
    max_device = max(ssd_devices, key=ssd_devices.get)
    
    return max_device


# def merge_files(workers_url: Dict[str, str], chunks: List[Dict], probe_results: Union[Dict[str, Dict[str, float]], None], memory_bnd_bytes: float, block_size: int = 0):
#     avg_sample_size = []
#     sockets = {}
#     context = zmq.Context()
#     for chunk in chunks:
#         worker_addr = workers_url[chunk['Location']]
#         if workers_url not in sockets:
#             socket = context.socket(zmq.REQ)
#             socket.connect(f"tcp://{worker_addr}")
#             sockets[workers_url[chunk['Location']]] = socket
        
#         req = pb.InspectChunkRequest()
#         req.chunk_path = f"{ssd_dir}/local/{chunk['ETag']}/"
#         sockets[worker_addr].send_multipart([b"inspect_chunk", req.SerializeToString()])
#         data = sockets[worker_addr].recv()
#         resp = pb.InspectChunkResponse.FromString(data)
#         avg_sample_size.append(resp.total_size / resp.file_count)
#     avg_sample_size = np.mean(avg_sample_size)
    
#     # Prometheus服务器的地址
#     url = "http://prometheus-service.monitoring.svc.cluster.local:8080/api/v1/query"
    
#     def objective(logBL): 
#         job_wait_time = 0
#         BL = math.ceil((2**logBL) / avg_sample_size)

#         for worker_probe_rlt in probe_results[obj_job]:
#             worker = worker_probe_rlt['worker']
#             job_id = '-'.join(worker.split('-')[:-1])
#             worker_idx = int(worker.split("-")[-1])
#             probe_perf = worker_probe_rlt['values']
#             B = probe_perf['BatchSize']
#             logB = np.log2(B * avg_sample_size)
#             W = probe_perf['NumDataLoadingWorkers']
#             C = probe_perf['NumCores']
#             PT =  probe_perf['ProcessingTime']
#             DT = probe_perf['DataLoadingTime']
#             CT = probe_perf['ComputingTime']
        
#             worker_on_node = dlworkers_nodes[worker_idx]
#             filter_chunks = [chunk for chunk in chunks if job_id in chunk['Jobs']]
#             num_local_files, num_remote_files = 0, 0
#             for chunk in filter_chunks:
#                 dir = f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}"
#                 num_files = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
#                 if chunk['Location'] in worker_on_node:
#                     num_local_files += num_files
#                 else:
#                     num_remote_files += num_files

#             if num_local_files > 0:
#                 input = [logBL, logB]
#                 features = ['block_size', 'total_size']
#                 input = pd.DataFrame([input], columns=features)
#                 with open('/app/ioperf/models/rf_local.pkl', 'rb') as f:
#                     local_io_regressor = pkl.load(f)
#                 pred_local_batch_io_time = local_io_regressor.predict(input)[0]

#             pred_remote_batch_io_time = 0
#             if num_remote_files > 0:
#                 # 获取worker所在节点上主网卡的网络流量
#                 command = f'ssh {getpass.getuser()}@{worker_on_node} "ip route get 1 | awk \'{{print $5;exit}}\'"'
#                 interface_name = os.popen(command).read().strip()

#                 query = "irate(node_network_transmit_bytes_total{device='%s', node_ip='%s'}[5m]) + irate(node_network_receive_bytes_total{device='kube-bridge', node_ip='%s'}[5m])" % (interface_name, worker_on_node, worker_on_node)
#                 response = requests.get(url, params={'query': query})
#                 result = json.loads(response.content.decode('utf-8'))['data']['result'][0]

#                 # 计算网络的可用带宽（Bps)
#                 available_bandwidth = MAX_BANDWIDTH - float(result['value'][1])
#                 available_bandwidth = (available_bandwidth * 8) /(1024**3)  # convert to Gbps
#                 logger.info(f"Available bandwidth for {result['metric']['node_ip']}: {available_bandwidth} Gps")

#                 input = [logBL, logB, available_bandwidth]
#                 features = ['block_size', 'total_size', 'bandwidth']
#                 input = pd.DataFrame([input], columns=features)
#                 with open('/app/ioperf/models/rf_remote.pkl', 'rb') as f:
#                     remote_io_regressor = pkl.load(f)
#                 pred_remote_batch_io_time = remote_io_regressor.predict(input)[0]
            
#             # 总IO时间为本地IO和远程IO的调和平均
#             local_ratio = num_local_files / (num_local_files + num_remote_files)
#             I = pred_local_batch_io_time * local_ratio + pred_remote_batch_io_time * (1 - local_ratio)
#             est_p1 = W * PT / C
#             if B >= BL:
#                 s = B % BL
#             else:
#                 s = BL - B
#             est_p2 = min((s * PT * W) / (C * B), (W-1) * (DT + CT))
#             est_process_time = max(est_p1 - est_p2, 0)
            
#             # compute total waiting time
#             wait_time = max(I + est_process_time - CT, 0)
#             job_wait_time = max(job_wait_time, wait_time)
            
#             with open(f"/mnt/nfs/ssd/local/{worker}", 'a+') as f:
#                 log_txt = f"batch_size: {B}, block_size: {BL}, log2(block_size): {logBL}, I: {I}, est_process_time: {est_process_time}, est_p1: {est_p1}, est_p2: {est_p2}, computing_time: {CT}, est_job_wait_time: {job_wait_time}"
#                 f.write(f"{log_txt}\n")
#                 logger.info(log_txt)

#         return job_wait_time
    
#     # determin the optimal block_size
#     if block_size == 0:
#         lower_bound = np.log2(avg_sample_size)
#         upper_bound = np.log2(min(memory_bnd_bytes, min([chunk['ChunkSize'] for chunk in chunks])))
#         opt_logBL = []
#         for obj_job in probe_results:
#             job_info = mongo_operator.find_job(obj_job)
#             dlworkers_nodes = job_info['Nodes']
#             res = minimize_scalar(objective, bounds=(lower_bound, upper_bound), options=dict(xatol=1e-9, maxiter=200, disp=3))
#             opt_logBL.append(res.x)
#         opt_block_size = int(np.round(2**np.mean(opt_logBL) / avg_sample_size))
#     else:
#         opt_block_size = block_size

#     with open(f"/mnt/nfs/ssd/local/opt_block_size", 'a+') as f:
#         log_txt = f"block_size: {opt_block_size}, block_size_bytes: {opt_block_size * avg_sample_size}"
#         f.write(f"{log_txt}\n")
#         logger.info(log_txt)
        
#     for i in range(len(chunks)):
#         chunk = chunks[i]
#         flag = True
#         if 'Blocks' in chunk:
#             for block in chunk['Blocks']:
#                 if not os.path.exists(f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}/{block['Name']}"):
#                     flag = False
#             if flag:
#                 continue

#         worker_addr = workers_url[chunk['Location']]
#         req = pb.MergeFileRequest()
#         req.etag = chunk['ETag']
#         req.blockSize = opt_block_size
#         sockets[worker_addr].send_multipart([b"merge", req.SerializeToString()])
#         data = sockets[worker_addr].recv()
#         resp = pb.MergeFileResponse.FromString(data)
#         if not resp.rc:
#             logger.error(f"Failed to merge small files for chunk {chunks[i]['ETag']}: {resp.error}")
#         else:
#             chunks[i]['Blocks'] = [{"Name": block.name, "Length": block.length} for block in resp.blocks]
    
#     for worker_addr in sockets:
#         sockets[worker_addr].close()


def simulated_annealing(func, bounds, initial_temp, cooling_rate, iter_per_temp):
    current_pos = np.random.uniform(bounds[0], bounds[1])
    current_energy = func(current_pos)
    temp = initial_temp

    while temp > 0.1:  # 结束条件，例如冷却到一定温度
        for _ in range(iter_per_temp):
            new_pos = np.random.uniform(bounds[0], bounds[1])
            new_energy = func(new_pos)
            delta_energy = new_energy - current_energy

            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temp): 
                # 如果找到更好的解，或者满足Metropolis准则，则接受新解
                current_pos = new_pos
                current_energy = new_energy
                
        temp *= cooling_rate  # 降温

    return current_pos, current_energy


class GASolver(object):
    def __init__(self, fitness, bounds, pop_size=10, n_gen=10, mutation_rate=0.1):
        self.fitness = fitness
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        
    # 种群初始化
    def initialize_population(self):
        return np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=self.pop_size)

    # 选择操作
    @staticmethod
    def selection(pop, fitness):
        idx = np.random.choice(np.arange(len(pop)), size=2, replace=False, p=fitness/sum(fitness))
        return pop[idx]

    # 交叉操作.
    @staticmethod
    def crossover(parents):
        c = np.random.uniform(low=min(parents), high=max(parents), size=1)
        return c

    # 变异操作
    def mutation(self, child):
        if np.random.rand() < self.mutation_rate:
            child = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=1)
        return child

    # 遗传算法主循环
    def genetic_algorithm(self):
        pop = self.initialize_population()
        for gen in range(self.n_gen):
            new_pop = []
            for i in range(self.pop_size):
                parents = self.selection(pop, [self.fitness(p) for p in pop])
                child = self.crossover(parents)
                child = self.mutation(child)
                new_pop.append(child)
            pop = np.array(new_pop)
        
        best_individual = pop[np.argmin([self.fitness(p) for p in pop])]
        return best_individual



def merge_files(workers_url: Dict[str, str], chunks: List[Dict], probe_results: Union[Dict[str, Dict[str, float]], None], memory_bnd_bytes: float, block_size: int = 0):
    avg_sample_size = []
    sockets = {}
    context = zmq.Context()
    for chunk in chunks:
        worker_addr = workers_url[chunk['Location']]
        if workers_url not in sockets:
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{worker_addr}")
            sockets[workers_url[chunk['Location']]] = socket
        
        req = pb.InspectChunkRequest()
        req.chunk_path = f"{ssd_dir}/local/{chunk['ETag']}/"
        sockets[worker_addr].send_multipart([b"inspect_chunk", req.SerializeToString()])
        data = sockets[worker_addr].recv()
        resp = pb.InspectChunkResponse.FromString(data)
        avg_sample_size.append(resp.total_size / resp.file_count)
    avg_sample_size = np.mean(avg_sample_size)
    
    # Prometheus服务器的地址
    url = "http://prometheus-service.monitoring.svc.cluster.local:8080/api/v1/query"
    
    def objective(logBL):

        BL = math.ceil((2**logBL) / avg_sample_size)

        for worker_probe_rlt in probe_results[obj_job]:
            worker = worker_probe_rlt['worker']
            job_id = '-'.join(worker.split('-')[:-1])
            worker_idx = int(worker.split("-")[-1])
            probe_perf = worker_probe_rlt['values']
            B = probe_perf['BatchSize']
            logB = np.log2(B * avg_sample_size)
            W = probe_perf['NumDataLoadingWorkers']
            C = probe_perf['NumCores']
            PT =  probe_perf['ProcessingTime']
            DT = probe_perf['DataLoadingTime']
            CT = probe_perf['ComputingTime']
        
            worker_on_node = dlworkers_nodes[worker_idx]
            filter_chunks = [chunk for chunk in chunks if job_id in chunk['Jobs']]
            num_local_files, num_remote_files = 0, 0
            for chunk in filter_chunks:
                dir = f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}"
                num_files = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
                if chunk['Location'] in worker_on_node:
                    num_local_files += num_files
                else:
                    num_remote_files += num_files

            if num_local_files > 0:
                input = [logBL, logB]
                features = ['block_size', 'total_size']
                input = pd.DataFrame([input], columns=features)
                with open('/app/ioperf/models/rf_local.pkl', 'rb') as f:
                    local_io_regressor = pkl.load(f)
                pred_local_batch_io_time = local_io_regressor.predict(input)[0]

            pred_remote_batch_io_time = 0
            if num_remote_files > 0:
                # 获取worker所在节点上主网卡的网络流量
                command = f'ssh {getpass.getuser()}@{worker_on_node} "ip route get 1 | awk \'{{print $5;exit}}\'"'
                interface_name = os.popen(command).read().strip()

                query = "irate(node_network_transmit_bytes_total{device='%s', node_ip='%s'}[5m]) + irate(node_network_receive_bytes_total{device='kube-bridge', node_ip='%s'}[5m])" % (interface_name, worker_on_node, worker_on_node)
                response = requests.get(url, params={'query': query})
                result = json.loads(response.content.decode('utf-8'))['data']['result'][0]

                # 计算网络的可用带宽（Bps)
                available_bandwidth = MAX_BANDWIDTH - float(result['value'][1])
                available_bandwidth = (available_bandwidth * 8) /(1024**3)  # convert to Gbps
                logger.info(f"Available bandwidth for {result['metric']['node_ip']}: {available_bandwidth} Gps")

                input = [logBL, logB, available_bandwidth]
                features = ['block_size', 'total_size', 'bandwidth']
                input = pd.DataFrame([input], columns=features)
                with open('/app/ioperf/models/rf_remote.pkl', 'rb') as f:
                    remote_io_regressor = pkl.load(f)
                pred_remote_batch_io_time = remote_io_regressor.predict(input)[0]
            
            local_ratio = num_local_files / (num_local_files + num_remote_files)
            I = pred_local_batch_io_time * local_ratio + pred_remote_batch_io_time * (1 - local_ratio)
            
            PPT = (PT*W)/C
            B_ = (BL-B%BL) if B > BL else BL-B
            obj = max((2*I + PPT - 2*CT*(W-1)) / (2*W-1), 0)
            cond1 = ((PPT+I)/(W-1)) - CT
            cond2 = ((((B_+B)*(PT*W)/(C*B)) + I) / (W-1)) - CT
            print(f"obj: {obj}, cons1: {cond1}, cons2: {cond2}")
            if obj > cond1:
                if obj > cond2:
                    obj = (I + PPT(1-B_/B) - (W-1)*CT) / W
            else:
                obj = (I + PPT - (W-1)*CT) / W
            
            log_txt = f"batch_size: {B}, block_size: {BL}, log2(block_size): {logBL}, I: {I}, computing_time: {CT}, waiting_time: {obj}"
            logger.info(log_txt)
            # with open(f"/mnt/nfs/ssd/local/{worker}", 'a+') as f:
            #     f.write(f"{log_txt}\n")

        return max(obj, 0)
    
    # determin the optimal block_size
    if block_size == 0:
        lower_bound = np.log2(avg_sample_size)
        # lower_bound = 20
        upper_bound = np.log2(min(memory_bnd_bytes, min([chunk['ChunkSize'] for chunk in chunks])))
        opt_logBL = []
        for obj_job in probe_results:
            job_info = mongo_operator.find_job(obj_job)
            dlworkers_nodes = job_info['Nodes']
            
            # ga = GASolver(objective, bounds=(lower_bound, upper_bound))
            # logBL = ga.genetic_algorithm()
            # opt_logBL.append(logBL)
            
            logBL, waiting_time  = simulated_annealing(objective, bounds=(lower_bound, upper_bound), initial_temp = 50, cooling_rate = 0.9, iter_per_temp = 10)
            opt_logBL.append(logBL)
            
            # res = minimize_scalar(objective, bounds=(lower_bound, upper_bound), options=dict(xatol=1e-30, maxiter=200))
            # opt_logBL.append(res.x)
        opt_block_size = int(np.ceil(2**np.mean(opt_logBL) / avg_sample_size))
        with open(f"/mnt/nfs/ssd/local/opt_block_size", 'a+') as f:
            log_txt = f"block_size: {opt_block_size}, block_size_bytes: {opt_block_size * avg_sample_size}"
            f.write(f"{log_txt}\n")
            logger.info(log_txt)
    else:
        opt_block_size = block_size
        
    for i in range(len(chunks)):
        chunk = chunks[i]
        flag = True
        if 'Blocks' in chunk:
            for block in chunk['Blocks']:
                if not os.path.exists(f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}/{block['Name']}"):
                    flag = False
            if flag:
                continue

        worker_addr = workers_url[chunk['Location']]
        req = pb.MergeFileRequest()
        req.etag = chunk['ETag']
        req.blockSize = opt_block_size
        sockets[worker_addr].send_multipart([b"merge", req.SerializeToString()])
        data = sockets[worker_addr].recv()
        resp = pb.MergeFileResponse.FromString(data)
        if not resp.rc:
            logger.error(f"Failed to merge small files for chunk {chunks[i]['ETag']}: {resp.error}")
        else:
            chunks[i]['Blocks'] = [{"Name": block.name, "Length": block.length} for block in resp.blocks]
    
    for worker_addr in sockets:
        sockets[worker_addr].close()


# The func will make job/worker placement decision and download data on HDD.
def scheduler(workers_url: Dict[str, str], job_queue: Dict[str, JobQueueItem]):
    while True:
        # sort by ssd gap first then hdd gap
        job_queue_items = sorted(job_queue.items(), key=lambda item: (item[1].storage_gap[1], item[1].storage_gap[0]))
        for job_id, job in job_queue_items:
            if job.deployable:
                continue
            
            jobs = [job_queue[job_id] for job_id in job.peers]
            if job.storage_gap[0] > 0:
                # The dataset for job has not been downloaded and it's storage gap on SSD has not been computed.
                # So this step will try to download the dataset on HDD. 
                # If the returned gap is still inf, it means this dataset even can't be placed on HDD.
                gap = eval_resource_gap(workers_url, jobs, download=True)
            else: # no HDD storage presure
                if job.storage_gap[1] > 0:
                    # The dataset has been downloaded on HDD, but has not been extracted on SSD.
                    gap = eval_resource_gap(workers_url, jobs, download=False)
                else:
                    if len(job.chunks) > 0:
                        continue

            # Broadcast the information to jobs that are using the same dataset (gang-scheduling), avoid re-downloading dataset
            for i, job_id in enumerate(job.peers):
                obj = job_queue[job_id]
                obj.deployable = (gap[0] + gap[1] == 0)
                obj.storage_gap = gap
                obj.chunks = jobs[0].chunks
                obj.nodes = jobs[i].nodes
                job_queue[job_id] = obj

        time.sleep(schedule_freq)


def executer(workers_url: Dict[str, str], job_queue: Dict[str, JobQueueItem], merging: bool = True, block_size: int = 0):
    while True:
        if len(job_queue) > 0:
            job_id, job = sorted(job_queue.items(), key=lambda item: (item[1].storage_gap[1], item[1].storage_gap[0]))[0]
            if job.deployable:
                # the extraction step will extract data from HDD layer to SSD
                extract_chunks(workers_url, job.chunks)
                
                peer_jobs = [job_queue[job_id] for job_id in job.peers]
                jobs_full_name = [f"{job.dltdeploy_id}-{job.job_id}" for job in peer_jobs]
                if merging:
                    save_chunks_info(jobs_full_name, job.chunks)
                    
                    if block_size == 0:
                        probe_rlt = probe(jobs=peer_jobs, post=True)
                        if probe_rlt is None:
                            continue
                        merge_files(workers_url, job.chunks, probe_rlt, memory_bnd_bytes=1e9, block_size=block_size)
                    else:
                        merge_files(workers_url, job.chunks, None, memory_bnd_bytes=1e9, block_size=block_size)

                    for peer_job_id in job.peers:
                        obj = job_queue[peer_job_id]
                        obj.chunks = job.chunks
                        job_queue[peer_job_id] = obj

                    # add 'Blocks' field for each chunk
                    for chunk in job.chunks:
                        mongo_operator.update_chunk(chunk['ETag'], {'$set': {"Blocks": chunk['Blocks']}})
                else:
                    save_chunks_info(jobs_full_name, job.chunks)

                # all peers must be deployable because they are using the same dataset
                peer_jobs = [job_queue[job_id] for job_id in job.peers]
                if k8s_operator.deploy(peer_jobs):
                    for peer_id in job.peers:
                        peer = job_queue[peer_id]
                        mongo_operator.update_job(f"{peer.dltdeploy_id}-{peer.job_id}", {"Status": "ready"})
                        del job_queue[peer_id]
                else:
                    logger.error(f'Failed to deploy DLTDeployment {job.dltdeploy_id}.')
        time.sleep(schedule_freq)

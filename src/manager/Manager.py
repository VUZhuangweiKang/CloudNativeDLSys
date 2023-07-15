import os.path
import time
import multiprocessing as mp
import concurrent.futures
import numpy as np
import pickle as pkl
from typing import List, Dict, Union, Tuple
import zmq
import getpass
import random
import glob
import requests
import psutil
from itertools import cycle
import json
import faulthandler
import pandas as pd
import copy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sys
sys.path.append("../")
from manager import *
from manager.Placement import *
from manager.ChunkOps import *

faulthandler.enable()


def eval_resource_gap(workers_url, jobs: List[JobQueueItem], download=True, schedule_alg = None, data_place_alg = None) -> Tuple[float]:
    gang = len(jobs) > 1
    job = jobs[0]
    distributed = job.spec['numWorkers'] > 1
    cred, bucket = job.cred, job.spec['datasource']['bucket']
    resources = k8s_operator.get_free_resources(hdd_dir=hdd_dir, ssd_dir=ssd_dir)
    # print('------------ resources before schedule ------------')
    # print(resources)

    preemptive_chunks_on_hdd = cost_aware_lrfu(ssd=False)
    
    if gang:
        hdd_gap = multi_job_placement(resources, preemptive_chunks_on_hdd, jobs, ssd=False, scheduler=schedule_alg, data_place_alg=data_place_alg)
    elif distributed:
        hdd_gap = multi_worker_placement(resources, preemptive_chunks_on_hdd, job, ssd=False, scheduler=schedule_alg, data_place_alg=data_place_alg)
    else:
        hdd_gap = base_placement(resources, preemptive_chunks_on_hdd, job, ssd=False, scheduler=schedule_alg, data_place_alg=data_place_alg)

    logger.info(f"HDD storage gap: {hdd_gap}")
    
    # can't meet computation resource requirement or some ssd chunks can't be downloaded
    if hdd_gap is None:
        return np.inf, np.inf
    elif hdd_gap > 0:
        return hdd_gap, np.inf

    if download:
        context = zmq.Context()
        s3_operator = S3Operator(mongo_operator.get_s3auth(**cred))
        chunks = []
        
        chunk_groups = defaultdict(list)
        for chunk in jobs[0].chunks:
            if not chunk['ExistOnHDD']:
                chunk_groups[chunk['SourceLocation']].append(chunk)
            else:
                logger.info(f"skip chunk {chunk['ETag']}")
                chunks.append(chunk)
        
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for node in chunk_groups:
                futures.append(executor.submit(download_chunks, workers_url[node], chunk_groups[node], s3_operator, bucket, context))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                chunks.extend(result)
                
        for i in range(len(jobs)):
            jobs[i].chunks = chunks

    resources = k8s_operator.get_free_resources(hdd_dir=hdd_dir, ssd_dir=ssd_dir)
    
    # Now, we have downloaded ssd chunks and know the actual chunk size. So the returned gap value is
    # based on the extracted chunk size. This step will also update the chunk location if the node is unable to
    # store the extracted chunks.
    preemptive_chunks_on_ssd = cost_aware_lrfu(ssd=True)
    if gang:
        ssd_gap = multi_job_placement(resources, preemptive_chunks_on_ssd, jobs, ssd=True, scheduler=schedule_alg, data_place_alg=data_place_alg)
    elif distributed:
        ssd_gap = multi_worker_placement(resources, preemptive_chunks_on_ssd, job, ssd=True, scheduler=schedule_alg, data_place_alg=data_place_alg)
    else:
        ssd_gap = base_placement(resources, preemptive_chunks_on_ssd, job, ssd=True, scheduler=schedule_alg, data_place_alg=data_place_alg)
    
    # save nodes information into database
    for job in jobs:
        mongo_operator.update_job(f"{job.dltdeploy_id}-{job.job_id}", update={"Nodes": job.nodes})

    logger.info(f"SSD storage gap: {ssd_gap}")
    
    # print('------------ resources after schedule ------------ ')
    # print(resources)
    
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


def merge_files(jobs: List[JobQueueItem], chunks: List[Dict], workers_url: Dict[str, str], fix_block_size: int = None):
    jobs_full_name = [f"{job.dltdeploy_id}-{job.job_id}" for job in jobs]
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
    chunks_iterator = cycle(chunks)
    chunk_idx = 0
    history = []
    
    def merge(merge_chunks, block_size, num_samples=None):
        for i in range(len(merge_chunks)):
            chunk = merge_chunks[i]
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
            req.blockSize = int(block_size)
            if num_samples is not None:
                req.numSamples = int(num_samples[i])
            else:
                req.numSamples = -1
            sockets[worker_addr].send_multipart([b"merge", req.SerializeToString()])
            data = sockets[worker_addr].recv()
            resp = pb.MergeFileResponse.FromString(data)
            if not resp.rc:
                logger.error(f"Failed to merge small files for chunk {chunk['ETag']}: {resp.error}")
            else:
                merge_chunks[i]['Blocks'] = [{"Name": block.name, "Length": block.length} for block in resp.blocks]
                
        save_chunks_info(jobs_full_name, merge_chunks)
        for job in jobs:
            job.chunks = merge_chunks
    
    def delete_blocks(merge_chunks):
        for chunk in merge_chunks:
            chunk_path = f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}"
            # if os.path.exists(chunk_path):
            #     shutil.rmtree(chunk_path)
            assert len(glob.glob(f"{chunk_path}/merged*")) > 0
            for block in glob.glob(f"{chunk_path}/merged*"):
                os.remove(block)
            if 'Blocks' in chunk:
                del chunk['Blocks']
                mongo_operator.replace_chunk(chunk['ETag'], chunk)
    
    class StopSearch(Exception):
        def __init__(self, *args: object, block_size: int, threadpool_size: int, loss: float) -> None:
            super().__init__(*args)
            self.block_size = block_size
            self.threadpool_size = threadpool_size
            self.loss = loss
            
    def get_loading_time(threadpool_size=1):
        probe_results = probe(jobs=jobs, post=True, merge=True, threadpool_size=threadpool_size)
        avg_batch_time = []
        stop_probe = True
        for obj_job in probe_results:
            job_batch_time = 0
            for worker_probe_rlt in probe_results[obj_job]:
                probe_perf = worker_probe_rlt['values']
                DT = probe_perf['DataLoadingTime']
                CT = probe_perf['ComputingTime']
                stop_probe = stop_probe and (not probe_perf['HasDataStall'])
                job_batch_time = max(job_batch_time, DT + CT)
            avg_batch_time.append(job_batch_time)
        return np.mean(avg_batch_time), stop_probe
    
    def ternary_search(min_size, max_size, tol_size=10, max_iters=10):
        num_iters = 0
        opt_completion_time = np.inf
        for chunk in chunks:
            if 'Blocks' in chunk:
                del chunk['Blocks']
                mongo_operator.replace_chunk(chunk['ETag'], chunk)
        
        while max_size - min_size > tol_size and num_iters < max_iters:
            mid1_size = int(min_size + (max_size - min_size) / 3)
            mid2_size = int(max_size - (max_size - min_size) / 3)
            
            merge_chunks = []
            
            # TODO: 暂定使用两个chunk，实际上样本数应该大于2*num_workers*prefetch_factor
            use_num_chunks = 2
            for _ in range(use_num_chunks):
                while True:
                    chk = next(chunks_iterator)
                    if chk['Category'] == 'train':
                        merge_chunks.append(chk)
                        break
            
            # merge_chunks = [chk for chk in chunks if chk['Category'] == 'train']
            if len(merge_chunks) == 0:
                return 1, 0
            
            def pipeline(merge_chunks, block_size):   
                nonlocal history     
                merge(merge_chunks, block_size)
                batch_completion_time = get_loading_time()
                logger.info(f"prob result: block_size: {block_size}, completion_time: {batch_completion_time}")
                delete_blocks(merge_chunks)
                extract_chunks(workers_url, merge_chunks)
                
                for chunk in merge_chunks:
                    mongo_operator.replace_chunk(chunk['ETag'], chunk)
                
                history.append([block_size, batch_completion_time])
                return batch_completion_time
    
            t1 = pipeline(merge_chunks, mid1_size)
            t2 = pipeline(merge_chunks, mid2_size)
            
            if t1 < t2:
                max_size = mid2_size
            else:
                min_size = mid1_size

            t = min(t1, t2)

            opt_completion_time = min(opt_completion_time, t)
            num_iters += 1

        return (min_size + max_size) / 2, opt_completion_time

    def objective_use_static_data(block_size):
        nonlocal history
        merge_chunks = []
        
        # Solution 2: always select the same subset
        use_num_chunks = sum([chk['Category'] == 'train' for chk in chunks])
        # use_num_chunks = 2
        for _ in range(use_num_chunks):
            while True:
                chk = next(chunks_iterator)
                if chk['Category'] == 'train':
                    merge_chunks.append(chk)
                    break

        merge(merge_chunks, block_size)
        batch_completion_time, stop_probe = get_loading_time()
        if stop_probe:
            raise StopSearch(block_size=block_size, loss=batch_completion_time)
        logger.info(f"prob result: block_size: {block_size}, completion_time: {batch_completion_time}")
        delete_blocks(merge_chunks)
        extract_chunks(workers_url, merge_chunks)
        
        for chunk in merge_chunks:
            mongo_operator.replace_chunk(chunk['ETag'], chunk)
        
        history.append([block_size, batch_completion_time])
        return batch_completion_time
    
    count_samples = lambda :sum([len(glob.glob(f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}/*")) for chunk in chunks if chunk['Category'] == 'train'])
    total_samples = count_samples()
    def objective_use_moving_data(params):
        nonlocal history, chunk_idx
        
        threadpool_size = params['threadpool_size']
        block_size = params['block_size']
        merge_chunks = []

        total_remaining_samples = count_samples()
        num_probe_samples = total_samples // 2
        # num_probe_samples = total_samples
        logger.info(f"num_probe_samples: {num_probe_samples}, total_remaining_samples: {total_remaining_samples}")
        
        num_samples = []
        while num_probe_samples > 0:
            chunk = chunks[chunk_idx]
            if chunk['Category'] == 'train':
                chunk_path = f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}"
                num_samples_in_chunk = len(glob.glob(f"{chunk_path}/*")) - len(glob.glob(f"{chunk_path}/merged_*.bin"))
                if num_samples_in_chunk == 0:
                    chunk_idx = (chunk_idx + 1) % len(chunks)
                    continue
                
                merge_chunks.append(chunk)
                num_samples.append(min(num_samples_in_chunk, num_probe_samples))
                logger.info(f"use {min(num_samples_in_chunk, num_probe_samples)}/{num_samples_in_chunk} samples from chunk {chunk_path}")
                num_probe_samples -= num_samples_in_chunk
            chunk_idx  = (chunk_idx + 1) % len(chunks)

        merge(merge_chunks, block_size, num_samples=num_samples)
        batch_completion_time, stop_probe = get_loading_time(threadpool_size)
        logger.info(f"prob result: block_size: {block_size}, threadpool_size: {threadpool_size}, completion_time: {batch_completion_time}")
        delete_blocks(merge_chunks)
        extract_chunks(workers_url, merge_chunks)
        
        for chunk in merge_chunks:
            mongo_operator.replace_chunk(chunk['ETag'], chunk)
        
        history.append([threadpool_size, block_size, batch_completion_time])
        
        if stop_probe:
            raise StopSearch(block_size=block_size, threadpool_size=threadpool_size, loss=batch_completion_time)
        
        return batch_completion_time
    
    def hyperopt_search(space, max_iters):
        trials = Trials()
        rlt = fmin(objective_use_moving_data, space, algo=tpe.suggest, trials=trials, max_evals=max_iters)
        return rlt, trials.best_trial['result']['loss']
    
    opt_block_size = 1
    opt_threadpool_size = 1
    do_merge = True
    if fix_block_size is None:
        try:
            # check if probing is needed, if probe is unnecessary, this will through StopSearch
            # otherwise, esecute the below steps
            init_batch_completion_time = objective_use_moving_data(params={'threadpool_size': 1, 'block_size': 1})
            
            threadpool_size_range = [1, 2, 4, 8]
            block_size_range = range(20, 320, 20)
            
            # threadpool_size_range = [1, 2, 3]
            # block_size_range = range(20, 120, 20)
            space = {
                'threadpool_size': hp.choice('threadpool_size', threadpool_size_range),
                'block_size': hp.choice('block_size', block_size_range)
            }
            max_iters = 10
            
            try:
                rlt, opt_completion_time = hyperopt_search(space, max_iters)
                opt_threadpool_size, opt_block_size = threadpool_size_range[rlt['threadpool_size']], block_size_range[rlt['block_size']]
            except StopSearch as ex:
                opt_threadpool_size, opt_block_size, opt_completion_time = ex.threadpool_size, ex.block_size, ex.loss
            
            # check if to reject search result
            if opt_completion_time > init_batch_completion_time:
                opt_threadpool_size = 1
                opt_block_size = 1
                do_merge = False
                # for exp only
                extract_chunks(workers_url, chunks)
            else:
                merge(merge_chunks=chunks, block_size=opt_block_size)

            logger.info(f"optimal configuration for job dltdeployment {jobs[0].dltdeploy_id}: block_size={opt_block_size}, threadpool_size={opt_threadpool_size}, comletion time: {opt_completion_time}")
        except StopSearch as ex:
            do_merge = False
            logger.info("merging is unnecessary...")
        finally:
            # for exp only
            with open('/mnt/nfs/hdd/local/opt_config', 'w+') as f:
                f.write(f"{opt_threadpool_size},{opt_block_size}")
            history = pd.DataFrame(history, columns=['threadpool_size', 'block_size', 'batch_completion_time'])
            history.to_csv(f"/mnt/nfs/hdd/local/{jobs[0].dltdeploy_id}.csv", index=False)
    else:
        merge(merge_chunks=chunks, block_size=fix_block_size)
    
    for worker_addr in sockets:
        sockets[worker_addr].close()
    
    return do_merge, opt_threadpool_size


# The func will make job/worker placement decision and download data on HDD.
def scheduler(workers_url: Dict[str, str], job_queue: Dict[str, JobQueueItem], deployable_job_queue: List[JobQueueItem], schedule_alg = None, data_place_alg = None):
    while True:
        while len(job_queue) > 0:
            # sort by ssd gap first then hdd gap
            job_queue_items = sorted(job_queue.items(), key=lambda item: (item[1].storage_gap[1], item[1].storage_gap[0]))
            job_id, job = job_queue_items[0]
            if job.deployable:
                continue
            
            jobs = [job_queue[job_id] for job_id in job.peers]
            if job.storage_gap[0] > 0:
                # The dataset for job has not been downloaded and it's storage gap on SSD has not been computed.
                # So this step will try to download the dataset on HDD. 
                # If the returned gap is still inf, it means this dataset even can't be placed on HDD.
                gap = eval_resource_gap(workers_url, jobs, download=True, schedule_alg=schedule_alg, data_place_alg=data_place_alg)
            else: # no HDD storage presure
                if job.storage_gap[1] > 0:
                    # The dataset has been downloaded on HDD, but has not been extracted on SSD.
                    gap = eval_resource_gap(workers_url, jobs, download=False, schedule_alg=schedule_alg, data_place_alg=data_place_alg)
                else:
                    if len(job.chunks) > 0:
                        continue

            # Broadcast the information to jobs that are using the same dataset (gang-scheduling), avoid re-downloading dataset
            deployable_jobs = []
            for job in jobs:
                obj = copy.deepcopy(job_queue[job.job_id])
                obj.deployable = (gap[0] + gap[1] == 0)
                obj.storage_gap = gap
                obj.chunks = jobs[0].chunks
                obj.nodes = job.nodes
                if obj.deployable:
                    deployable_jobs.append(obj)
                    del job_queue[job.job_id]
            if len(deployable_jobs) > 0:
                deployable_job_queue.put(deployable_jobs)
        
        time.sleep(schedule_freq)


def executer(workers_url: Dict[str, str], deployable_job_queue: List[JobQueueItem], merging: bool = True, block_size: int = 0, threadpool_size: int = 0):
    while True:
        deployable_jobs = deployable_job_queue.get(block=True)
        jobs_full_name = [f"{job.dltdeploy_id}-{job.job_id}" for job in deployable_jobs]
        first_job = deployable_jobs[0]
        
        # the extraction step will extract data from HDD layer to SSD
        extract_chunks(workers_url, first_job.chunks)
        save_chunks_info(jobs_full_name, first_job.chunks)
        if merging:
            if block_size == 0:
                do_merge, opt_threadpool_size = merge_files(deployable_jobs, first_job.chunks, workers_url)
            else:
                do_merge, opt_threadpool_size = merge_files(deployable_jobs, first_job.chunks, workers_url, fix_block_size=block_size)
            
            if threadpool_size != 0:
                opt_threadpool_size = threadpool_size

            for job in deployable_jobs:
                job.chunks = first_job.chunks
                job.spec['merge'] = int(do_merge)
                job.spec['probe'] = 0
                job.spec['threadpool_size'] = opt_threadpool_size

            for chunk in first_job.chunks:
                if do_merge:
                    mongo_operator.update_chunk(chunk['ETag'], {'$set': {"Blocks": chunk['Blocks']}})
                else:
                    mongo_operator.update_chunk(chunk['ETag'], {'$unset': {"Blocks": ""}})

        # all peers must be deployable because they are using the same dataset
        if k8s_operator.deploy(deployable_jobs):
            for job in deployable_jobs:
                mongo_operator.update_job(f"{job.dltdeploy_id}-{job.job_id}", {"Status": "ready"})
        else:
            logger.error(f'Failed to deploy DLTDeployment {first_job.dltdeploy_id}.')
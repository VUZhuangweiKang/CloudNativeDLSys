import numpy as np
import time
import zmq
from typing import List, Dict, Union, Tuple
from collections import defaultdict
import concurrent
import random
from copy import deepcopy
from manager import JobQueueItem, mongo_operator, S3Operator, logger, k8s_operator
from manager.ChunkOps import download_chunks, save_chunks_info, extract_chunks, delete_preemptive_chunks

random.seed(1234)


def probe(jobs: List[JobQueueItem], post=False, dataops_workers=None, merge=True, threadpool_size=1):
    
    def helper(job):
        if not k8s_operator.is_dltworker_running(job.dltdeploy_id, job.job_id):
            doc = mongo_operator.job_col.find_one(filter={'Meta.JobId': job.job_id, 'Meta.DLTDeployId': job.dltdeploy_id})
            if "Performance" in doc:
                return doc['Performance']
            else:
                return None
        else:
            return None

    results = {}
    # unprobed_jobs = []
    # for job in jobs:
    #     job_name = f"{job.dltdeploy_id}-{job.job_id}"
    #     result = helper(job)
    #     if result is None:
    #         unprobed_jobs.append(job)
    #     else:
    #         results[job_name] = result

    dltdeploy = jobs[0].dltdeploy_id
    if len(jobs) > 0:
        # the probe_job here is only for getting the s3auth and chunk information
        probe_job = jobs[0]
        bucket = probe_job.spec['datasource']['bucket']
        if not post:
            context = zmq.Context()
            s3_operator = S3Operator(mongo_operator.get_s3auth(**probe_job.cred))
            
            chunk_groups = defaultdict(list)
            for chunk in probe_job.chunks:
                chunk_groups[chunk['SourceLocation']].append(chunk)
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for node in chunk_groups:
                    futures.append(executor.submit(download_chunks, dataops_workers[chunk['SourceLocation']], chunk_groups[node], s3_operator, bucket, context))
            chunks = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    logger.error(f"Failed to download chunk {chunk['ETag']} for job {probe_job.job_id} in benefit aware dataset placement.")
                    return None
                else:
                    chunks.extend(result)
            probe_job.chunks = chunks

            extract_chunks(workers_url=dataops_workers, chunks=probe_job.chunks)
            jobs_full_name = [f"{dltdeploy}-{job.job_id}" for job in jobs]
            save_chunks_info(jobs_full_name, probe_job.chunks)

        for job in jobs:
            job_name = f"{dltdeploy}-{job.job_id}"
            job.chunks = probe_job.chunks
            job.spec['merge'] = int(merge)
            job.spec['probe'] = 1
            job.spec['threadpool_size'] = threadpool_size
            
        if not k8s_operator.deploy(jobs):
            logger.error(f"Failed to deploy dltdeployment {dltdeploy}.")
            return
        else:
            chunks_name = [chunk['ETag'] for chunk in probe_job.chunks if 'Blocks' in chunk]
            logger.info(f"probing dltdeployment {dltdeploy} using chunks {chunks_name}.")

        for job in jobs:
            while True:
                rlt = helper(job)
                if rlt is not None:
                    break
                time.sleep(5)
            results[job_name] = rlt
            mongo_operator.job_col.update_one(filter={'Meta.JobId': job.job_id, 'Meta.DLTDeployId': dltdeploy}, 
                                              update={"$unset": {"Performance": ""}})

        if not k8s_operator.delete_dltdeployment(dltdeploy):
            logger.error(f"failed to delete DLTDeployment {dltdeploy} when ending probing job {job.job_id}")
        else:
            logger.info(f"delete DLTDeployment {dltdeploy} when ending probing job {job.job_id}")
    return results


def sort_nodes_by_weights(resources, preemptive_chunks: Dict[str, List[Dict]], job: JobQueueItem, ssd: bool, gang: bool=False) -> List[Tuple]:
    # Principle: place as much data on the same node with the DL job as possible, but also avoid fragmenting nodes
    job.chunks.sort(key=lambda item: item['Size'], reverse=True)
    storage_dev = 'ssd' if ssd else 'hdd'
    
    # 1. 计算集群中所有可用空间：free_space + preemptive_space
    total_storage_capacity = 0.0
    nodes_list = list(resources['gpu'].keys())
    for node in nodes_list:
        if ssd:
            preemptive_space = sum([chunk["ChunkSize"] for chunk in preemptive_chunks[node]])
        else:
            preemptive_space = sum([chunk["Size"] for chunk in preemptive_chunks[node]])
        total_storage_capacity += preemptive_space + resources[storage_dev][node]

    # 2. 计算每个节点上的可用空间
    total_storage_space = defaultdict(float)
    for node, chunk_lst in preemptive_chunks.items():
        total_storage_space[node] += resources[storage_dev][node] + sum([chunk["ChunkSize"] if ssd else chunk['Size'] for chunk in chunk_lst])

    # 3. 计算每个节点上已经存在的chunks的大小，以及这个job的所需的存储空间
    total_exist_chunk_size = 0
    total_chunks_size = 0
    exist_chunks_on_node = defaultdict(float)
    for chunk in job.chunks:
        size = chunk['ChunkSize'] if ssd else chunk['Size']
        exist = chunk['ExistOnSSD'] if ssd else chunk['ExistOnHDD']
        if exist:
            loc = chunk['Location'] if ssd else chunk['SourceLocation']
            exist_chunks_on_node[loc] += size
            total_exist_chunk_size += size
        total_chunks_size += size
    remaining_chunks_size = total_chunks_size - total_exist_chunk_size
    if total_chunks_size == 0:
        remaining_chunks_ratio = 0
    else:
        remaining_chunks_ratio = remaining_chunks_size / total_chunks_size

    # 4. 计算每个node的weight
    weights = defaultdict(float)
    for node in nodes_list:
        # # term1: binary indicator to signify whether computation resource request can be satisfied
        # term1 = int(resources['gpu'][node] >= job.spec['resource_requests']['GPU'])
        # if gang:
        #     term1 *= (resources['gpu'][node]) / sum(list(resources['gpu'].values()))
        
        # since the cluster doesn't have GPU, so now we use CPU
        term1 = int(resources['cpu'][node] >= job.spec['resource_requests']['CPU'])
        if term1:
            # 1e6 is for avoiding the case: node1(1, 0, 0), node2: (0, 1, 0)
            term1 += 1e-6
        if gang:
            term1 *= (resources['cpu'][node]) / sum(list(resources['cpu'].values()))

        # term2: percentage of existing chunks on this node
        term2 = 0.0
        if node in exist_chunks_on_node:
            term2 = exist_chunks_on_node[node] / total_chunks_size

        # term3: capacity of deploying non-existing chunks on this node in percentage
        left_storage_capacity = min(total_storage_space[node], remaining_chunks_size)
        term3 = remaining_chunks_ratio * left_storage_capacity / total_storage_capacity

        weights[node] = term1 + term2 + term3
        logger.info(f"node weights for job {job.dltdeploy_id}-{job.job_id} on {node}: {term1} + {term2} + {term3} = {weights[node]}")
        

    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights


def bf_nodes_weights(resources) -> List[Tuple]:
    nodes_list = list(resources['gpu'].keys())
    weights = defaultdict(float)
    total_cpus = sum([resources['cpu'][node] for node in nodes_list])
    total_memory = sum([resources['memory'][node] for node in nodes_list])
    for node in nodes_list:
        weights[node] = resources['cpu'][node] / total_cpus + resources['memory'][node] / total_memory
    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights

def ff_nodes_weights(resources, job: JobQueueItem) -> List[Tuple]:
    nodes_list = list(resources['gpu'].keys())
    weights = {node: 0 for node in nodes_list}
    for node in nodes_list:
        if resources['cpu'][node] >= job.spec['resource_requests']['CPU'] and resources['memory'][node] >= job.spec['resource_requests']['Memory']:
            weights[node] += 1
            break
    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights


# Job and Data placement strategy for (1 job/ 1 worker) use case
def base_placement(resources, preemptive_chunks: Dict[str, List[Dict]], job: JobQueueItem, ssd: bool = True, worker_idx = None, gang: bool = False, scheduler = None, data_place_alg = None) -> Union[float, None]:
    storage_dev = 'ssd' if ssd else 'hdd'
    
    # although we have sorted the nodes list, it doesn't guarantee the job can be deployed
    if scheduler == 'ff':
        node_weights = ff_nodes_weights(resources, job)
    elif scheduler == "bf":
        node_weights = bf_nodes_weights(resources)
    else:
        node_weights = sort_nodes_by_weights(resources, preemptive_chunks, job, ssd, gang=gang)
    
    request_gpu = job.spec['resource_requests']['GPU']
    request_cpu = job.spec['resource_requests']['CPU']
    request_memory = job.spec['resource_requests']['Memory']
    
    placement = None
    for node, weight in node_weights:
        if resources['gpu'][node] >= request_gpu and resources['cpu'][node] >= request_cpu and resources['memory'][node] >= request_memory:
            resources['gpu'][node] -= request_gpu
            resources['cpu'][node] -= request_cpu
            resources["memory"][node] -= request_memory
            placement = node
            break

    if placement is None:
        return np.inf  # GPU resource is not enough for deploying the job
    else:
        job.nodes = [placement]
    
    # compute the storage gap when placing chunks
    storage_gap = 0
    for i in range(len(job.chunks)):
        chunk = job.chunks[i]
        s = chunk['ChunkSize'] if ssd else chunk['Size']
        exist = chunk['ExistOnSSD'] if ssd else chunk['ExistOnHDD']
        # we do nothing for existing chunks because migrating data is always expensive
        if not exist:
            if ssd:
                chunk['ExistOnSSD'] = True
            else:
                chunk['ExistOnHDD'] = True
                
            all_nodes = [item[0] for item in node_weights]
            if data_place_alg != 'random':
                # locality-aware data placement
                for j, node_weight in enumerate(node_weights):
                    node, weight = node_weight
                    if resources[storage_dev][node] < s:
                        # try to acquire more space by evicting preemptive chunks
                        preemptive_chunks_on_node = preemptive_chunks[node]
                        rc, released_space = delete_preemptive_chunks(preemptive_chunks_on_node, s - resources[storage_dev][node], ssd)
                        preemptive_chunks[node] = preemptive_chunks_on_node
                        resources[storage_dev][node] += released_space
                        if not rc:  # space is not enough even after deleting preemptive chunks
                            if j == len(node_weights) - 1:
                                storage_gap += s
                            continue
                    resources[storage_dev][node] -= s
                    if ssd:
                        chunk['Location'] = node
                    else:
                        chunk['SourceLocation'] = node
                    break
            else:
                sel_node = all_nodes[i % len(all_nodes)]
                if ssd:
                    chunk['Location'] = sel_node
                else:
                    chunk['SourceLocation'] = sel_node

        if worker_idx is not None:
            worker_name = f"{job.dltdeploy_id}-{job.job_id}-{worker_idx}"
            if 'Jobs' not in chunk:
                chunk['Jobs'] = []
            chunk['Jobs'].append(worker_name)

    # reset
    for chunk in job.chunks:
        chunk['ExistOnHDD'] = False
        chunk['ExistOnSSD'] = False
        
    return storage_gap


# Data placement strategy for (1 job / N worker) use case
def multi_worker_placement(resources,  preemptive_chunks: Dict[str, List[Dict]], 
                           job: JobQueueItem, ssd: bool = True, gang: bool = False, scheduler = None, data_place_alg = None) -> Union[float, None]:
    storage_gap = 0
    num_workers = job.spec['numWorkers']
    chunks_group = np.array_split(job.chunks, num_workers)
    workers_placement = []
    job.spec['numWorkers'] = 1
    for i in range(num_workers):
        job.chunks = chunks_group[i].tolist()
        gap = base_placement(resources, preemptive_chunks, job, ssd, worker_idx=i, gang=gang, scheduler=scheduler, data_place_alg=data_place_alg)
        chunks_group[i] = job.chunks
        workers_placement.extend(job.nodes)
        storage_gap += gap

    job.spec['numWorkers'] = num_workers
    job.chunks = [item for sublist in chunks_group for item in sublist]

    job.nodes = workers_placement
    return storage_gap


def multi_job_placement(resources, preemptive_chunks: Dict[str, List[Dict]], jobs: List[JobQueueItem], ssd: bool = True, scheduler = None, data_place_alg = None) -> Union[float, None]:
    if scheduler == 'dlsys':
        jobs.sort(key=lambda job: job.spec['numWorkers'])
    
    storage_gaps = []
    chunks_placement = None
    for i in range(len(jobs)):
        job = jobs[i]
        if i > 0:
            job.chunks = deepcopy(chunks_placement)
        storage_gap = multi_worker_placement(resources, preemptive_chunks, job, ssd, gang=(i==0), scheduler=scheduler, data_place_alg=data_place_alg)
        storage_gaps.append(storage_gap)
        logger.info(f'Job placement for {job.dltdeploy_id}-{job.job_id}: {job.nodes}')
        if i == 0 and storage_gap == 0:
            for chunk in job.chunks:
                chunk['ExistOnHDD'] = True
                chunk['ExistOnSSD'] = True
                logger.info(f"Data placement for dltdeploy {job.dltdeploy_id} -- {chunk['ETag']} location: {chunk.get('Location', None)}, source: {chunk.get('SourceLocation', None)}")
            chunks_placement = deepcopy(job.chunks)
        # for chunk in job.chunks:
        #     logger.info(f"verify job {job.dltdeploy_id}-{job.job_id} -- {chunk['ETag']} locations: {chunk.get('Location', None)}, source: {chunk.get('SourceLocation', None)}")

    # reset
    for job in jobs:
        for chunk in job.chunks:
            chunk['ExistOnHDD'] = False
            chunk['ExistOnSSD'] = False
            
    # since all jobs are sharing the same dataset, they should have same storage gap
    assert len(np.unique(storage_gaps)==1)
    return storage_gaps[0]
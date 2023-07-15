import pandas as pd
import numpy as np
import json
import random
import string
from datetime import datetime, timedelta
from collections import defaultdict
from copy import deepcopy
import argparse


def sort_nodes_by_weights(resources, job, ssd: bool, gang: bool = False):
    # Principle: place as much data on the same node with the DL job as possible, but also avoid fragmenting nodes
    job['chunks'].sort(key=lambda item: item['Size'], reverse=True)
    storage_dev = 'ssd' if ssd else 'hdd'

    # 1. 计算集群中所有可用空间：free_space + preemptive_space
    total_storage_capacity = 0.0
    nodes_list = list(resources['gpu'].keys())

    total_storage_space = defaultdict(float)
    for node in nodes_list:
        total_storage_space[node] = resources[storage_dev][node]
        total_storage_capacity += resources[storage_dev][node]

    # 3. 计算每个节点上已经存在的chunks的大小，以及这个job的所需的存储空间
    total_exist_chunk_size = 0
    total_chunks_size = 0
    exist_chunks_on_node = defaultdict(float)
    for chunk in job['chunks']:
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
        # since the cluster doesn't have gpu, so now we use cpu
        term1 = resources['gpu'][node] >= job['gpu'] and resources['cpu'][node] >= job['cpu']
        if term1:
            # 1e6 is for avoiding the case: node1(1, 0, 0), node2: (0, 1, 0)
            term1 += 1e-6
        if gang and term1:
            term1 *= (resources['gpu'][node]) / sum(list(resources['gpu'].values()))

        # term2: percentage of existing chunks on this node
        term2 = 0.0
        if node in exist_chunks_on_node:
            term2 = exist_chunks_on_node[node] / total_chunks_size

        # term3: capacity of deploying non-existing chunks on this node in percentage
        left_storage_capacity = min(total_storage_space[node], remaining_chunks_size)

        # print(remaining_chunks_ratio, left_storage_capacity, total_storage_space[node], total_storage_capacity)
        term3 = remaining_chunks_ratio * left_storage_capacity / total_storage_capacity

        weights[node] = term1 + term2 + term3
        # print(f"node weights for job {job['dltdeploy_id']}-{job['job_id']} on {node}: {term1} + {term2} + {term3} = {weights[node]}")

    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights


def bf_nodes_weights(resources):
    nodes_list = list(resources['gpu'].keys())
    weights = defaultdict(float)
    total_cpus = sum([resources['cpu'][node] for node in nodes_list])
    total_gpus = sum([resources['gpu'][node] for node in nodes_list])
    for node in nodes_list:
        if total_cpus == 0:
            term1 = 0
        else:
            term1 = resources['cpu'][node] / total_cpus
            
        if total_gpus == 0:
            term2 = 0
        else:
            term2 = resources['gpu'][node] / total_gpus
        
        weights[node] = term1 + term2
    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights


def ff_nodes_weights(resources, job):
    nodes_list = list(resources['gpu'].keys())
    weights = {node: 0 for node in nodes_list}
    for node in nodes_list:
        if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu']:
            weights[node] += 1
            break
    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights


# Job and Data placement strategy for (1 job/ 1 worker) use case
def base_placement(resources, job, ssd: bool = True, worker_idx=None, gang: bool = False, scheduler=None, data_place_alg=None):
    storage_dev = 'ssd' if ssd else 'hdd'

    # although we have sorted the nodes list, it doesn't guarantee the job can be deployed
    if scheduler == 'ff':
        node_weights = ff_nodes_weights(resources, job)
    elif scheduler == "bf":
        node_weights = bf_nodes_weights(resources)
    else:
        node_weights = sort_nodes_by_weights(resources, job, ssd, gang=gang)

    request_gpu = job['gpu']
    request_cpu = job['cpu']

    placement = None
    for node, weight in node_weights:
        assert node is not None
        if resources['gpu'][node] >= request_gpu and resources['cpu'][node] >= request_cpu:
            resources['gpu'][node] -= request_gpu
            resources['cpu'][node] -= request_cpu
            placement = node
            break

    if placement is None:
        return np.inf  # computation resource is not enough for deploying the job
    else:
        job['location'] = [placement]

    # compute the storage gap when placing chunks
    resource_gap = 0
    for i in range(len(job['chunks'])):
        chunk = job['chunks'][i]
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
                flag = False
                for node, weight in node_weights:
                    if resources[storage_dev][node] >= s:
                        resources[storage_dev][node] -= s
                        if ssd:
                            chunk['Location'] = node
                        else:
                            chunk['SourceLocation'] = node
                        flag = True
                        break
                if not flag:
                    resource_gap += chunk['ChunkSize']
            else:
                # TODO: selected node might can't hold the chunk
                sel_node = all_nodes[i % len(all_nodes)]
                if ssd:
                    chunk['Location'] = sel_node
                else:
                    chunk['SourceLocation'] = sel_node

        if worker_idx is not None:
            worker_name = f"{job['dltdeploy_id']}-{job['job_id']}-{worker_idx}"
            if 'Jobs' not in chunk:
                chunk['Jobs'] = []
            chunk['Jobs'].append(worker_name)

    # reset
    # for chunk in job['chunks']:
    #     chunk['ExistOnHDD'] = False
    #     chunk['ExistOnSSD'] = False

    return resource_gap


# Data placement strategy for (1 job / N worker) use case
def multi_worker_placement(resources, job, ssd: bool = True, gang: bool = False, scheduler=None, data_place_alg=None):
    resource_gap = 0
    num_workers = job['numWorkers']
    chunks_group = np.array_split(job['chunks'], num_workers)
    workers_placement = []
    job['numWorkers'] = 1
    for i in range(num_workers):
        job['chunks'] = chunks_group[i].tolist()
        gap = base_placement(resources, job, ssd, worker_idx=i, gang=gang, scheduler=scheduler, data_place_alg=data_place_alg)
        chunks_group[i] = job['chunks']
        workers_placement.extend(job['location'])
        if gap > 0:
            break
    
    # revoke resources assigned to previous deployable workers
    if resource_gap > 0:
        print(workers_placement)
        for worker in workers_placement:
            resources['cpu'][worker] += job['cpu']
            resources['gpu'][worker] += job['gpu']
            
    job['numWorkers'] = num_workers
    job['chunks'] = [item for sublist in chunks_group for item in sublist]
    job['location'] = workers_placement
    return resource_gap


def multi_job_placement(resources, jobs, ssd: bool = True, scheduler=None, data_place_alg=None):
    if scheduler == 'ours':
        jobs.sort(key=lambda job: job['numWorkers'])

    resource_gaps = []
    chunks_placement = None
    for i, job in enumerate(jobs):
        if i > 0:
            job['chunks'] = deepcopy(chunks_placement)

        resource_gap = multi_worker_placement(resources, job, ssd, gang=(i == 0), scheduler=scheduler, data_place_alg=data_place_alg)    
        resource_gaps.append(resource_gap)
        if resource_gap > 0:
            break

        if i == 0 and resource_gap == 0:
            for chunk in job['chunks']:
                chunk['ExistOnHDD'] = True
                chunk['ExistOnSSD'] = True
                # print(f"Data placement for dltdeploy {job['dltdeploy_id']} -- {chunk['ETag']} location: {chunk.get('Location', None)}, source: {chunk.get('SourceLocation', None)}")
            chunks_placement = deepcopy(job['chunks'])

        # for chunk in job.chunks:
        #     logger.info(f"verify job {job.dltdeploy_id}-{job.job_id} -- {chunk['ETag']} locations: {chunk.get('Location', None)}, source: {chunk.get('SourceLocation', None)}")

    # reset
    # for job in jobs:
    #     for chunk in job['chunks']:
    #         chunk['ExistOnHDD'] = False
    #         chunk['ExistOnSSD'] = False

    # since all jobs are sharing the same dataset, they should have same storage gap
    return np.max(resource_gaps)


def generate_random_ip():
    octets = [random.randint(0, 255) for _ in range(4)]
    ip_address = '.'.join(map(str, octets))
    return ip_address


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sch", type=str, choices=['ours', 'ff', 'bf'], default='ours')
    parser.add_argument("--cluster", type=str, choices=['venus', 'earth', 'saturn', 'uranus'], default='venus')
    args = parser.parse_args()
    
    base_dir = f'./clusters/{args.cluster}/' 
    with open(f'{base_dir}/{args.cluster}.json', 'r') as f:
        trace = json.load(f)
    trace.sort(key=lambda x: x['submit_time'])
    datasets = {}
    cluster = {}
    results = []
    
    total_nodes = {'venus': 133, 'uranus': 264, 'earth': 143, 'saturn': 262}
    num_cpus = {'venus': 48, 'uranus': 64, 'earth': 48, 'saturn': 64}
    nodes = [generate_random_ip() for _ in range(total_nodes[args.cluster])]
    for res, s in [('gpu', 8), ('cpu', num_cpus[args.cluster]), ('ssd', np.inf)]:
        cluster[res] = {}
        for node in nodes:
            cluster[res][node] = s

    running_jobs = defaultdict(dict)

    lidx = 0
    while lidx < len(trace):
        log = deepcopy(trace[lidx])
        # check whether to release resources
        start_time = datetime.strptime(log['submit_time'], "%Y-%m-%d %H:%M:%S")

        for deploy in list(running_jobs.keys()):
            unfinished_jobs = []
            for i, hist_job in enumerate(running_jobs[deploy]):
                # print(start_time, hist_job['end_time'])
                if start_time > datetime.strptime(hist_job['end_time'], "%Y-%m-%d %H:%M:%S"):
                    for loc in hist_job['location']:
                        cluster['cpu'][loc] += hist_job['cpu']
                        cluster['gpu'][loc] += hist_job['gpu']
                        for chunk in hist_job['chunks']:
                            if chunk['Location'] == loc:
                                cluster['ssd'][chunk['Location']] += chunk['ChunkSize']
                else:
                    unfinished_jobs.append(i)

            if len(unfinished_jobs) == 0:
                results.append(running_jobs[deploy])
                del running_jobs[deploy]
            else:
                running_jobs[deploy] = [running_jobs[deploy][i] for i in unfinished_jobs]

        for job in log['jobs']:
            job['dltdeploy_id'] = log['deploy_id']
            job['chunks'] = log['datasource']['chunks']

        # schedule the job
        if len(log['jobs']) > 1:
            resource_gaps = multi_job_placement(cluster, log['jobs'], scheduler=args.sch)
        elif log['jobs'][0]['numWorkers'] > 1:
            resource_gaps = multi_worker_placement(cluster, log['jobs'][0], scheduler=args.sch)
        else:
            resource_gaps = base_placement(cluster, log['jobs'][0], worker_idx=0, scheduler=args.sch)

        if resource_gaps > 0:
            print(log)
            trace[lidx]['submit_time'] = (datetime.strptime(log['submit_time'], "%Y-%m-%d %H:%M:%S") + timedelta(seconds=100)).strftime("%Y-%m-%d %H:%M:%S")
            continue
        else:
            lidx += 1

        # print(log['jobs'])
        running_jobs[job['dltdeploy_id']] = log['jobs']
    
    with open(f'{base_dir}/{args.sch}.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # print(cluster)
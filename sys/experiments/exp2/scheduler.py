import pandas as pd
import numpy as np
import json
import random
import string
from datetime import datetime, timedelta
from collections import defaultdict
from copy import deepcopy
import argparse


def sort_nodes_by_weights(resources, job, ssd: bool, job_idx: bool = False):
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
        term1 = resources['gpu'][node] >= job['gpu'] and resources['cpu'][node] >= job['cpu']
        if term1:
            # 1e-9 is for avoiding the case: node1(1, 0, 0), node2: (0, 1, 0)
            term1 += 1e-9
        if job_idx == 0 and term1:
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


def wf_nodes_weights(resources, job, job_idx):
    nodes_list = list(resources['gpu'].keys())
    weights = defaultdict(float)
    filtered_nodes = []
    for node in nodes_list:
        if job_idx == 0:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu'] and resources['ssd'][node] >= sum([chunk['ChunkSize'] for chunk in job['chunks']]):
                filtered_nodes.append(node)
        else:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu']:
                filtered_nodes.append(node)
                
    total_cpus = sum([resources['cpu'][node] for node in filtered_nodes])
    total_gpus = sum([resources['gpu'][node] for node in filtered_nodes])
    total_space = sum([resources['ssd'][node] for node in filtered_nodes])
    
    for node in filtered_nodes:
        if total_cpus == 0:
            term1 = 0
        else:
            term1 = resources['cpu'][node] / total_cpus
            
        if total_gpus == 0:
            term2 = 0
        else:
            term2 = resources['gpu'][node] / total_gpus
        
        weights[node] = term1 + term2
        if job_idx == 0:
            weights[node] += resources['ssd'][node] / total_space
            
    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights

def bf_nodes_weights(resources, job, job_idx):
    weights = defaultdict(float)
    nodes_list = list(resources['gpu'].keys())
    filtered_nodes = []
    for node in nodes_list:
        if job_idx == 0:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu'] and resources['ssd'][node] >= sum([chunk['ChunkSize'] for chunk in job['chunks']]):
                filtered_nodes.append(node)
        else:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu']:
                filtered_nodes.append(node)
    total_cpus = sum([resources['cpu'][node] for node in filtered_nodes])
    total_gpus = sum([resources['gpu'][node] for node in filtered_nodes])
    total_space = sum([resources['ssd'][node] for node in filtered_nodes])
    
    for node in filtered_nodes:
        if total_cpus == 0:
            term1 = 0
        else:
            term1 = resources['cpu'][node] / total_cpus
            
        if total_gpus == 0:
            term2 = 0
        else:
            term2 = resources['gpu'][node] / total_gpus
        
        weights[node] = term1 + term2
        if job_idx == 0:
            weights[node] += resources['ssd'][node] / total_space
        
    weights = list(weights.items())
    weights.sort(key=lambda item: item[1])
    return weights


def ff_nodes_weights(resources, job, job_idx):
    nodes_list = list(resources['gpu'].keys())
    weights = {node: 0 for node in nodes_list}
    for node in nodes_list:
        if job_idx == 0:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu'] and resources['ssd'][node] >= sum([chunk['ChunkSize'] for chunk in job['chunks']]):
                weights[node] += 1
                break
        else:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu']:
                weights[node] += 1
                break
    weights = list(weights.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights


def csa_nodes_weights(resources, job, job_idx):
    nodes_list = list(resources['gpu'].keys())
    filter_nodes = []
    for node in nodes_list:
        if job_idx == 0:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu'] and resources['ssd'][node] >= sum([chunk['ChunkSize'] for chunk in job['chunks']]):
                filter_nodes.append(node)
        else:
            if resources['cpu'][node] >= job['cpu'] and resources['gpu'][node] >= job['gpu']:
                filter_nodes.append(node)

    exist_chunks_on_node = {node: 0 for node in filter_nodes}
    if len(exist_chunks_on_node) == 0:
        exist_chunks_on_node = {node: 0 for node in filter_nodes}
        weights = list(exist_chunks_on_node.items())
        random.shuffle(weights)
        return weights
    
    for chunk in job['chunks']:
        size = chunk['ChunkSize']
        exist = chunk['ExistOnSSD']
        if exist:
            loc = chunk['Location']
            if loc in filter_nodes:
                exist_chunks_on_node[loc] += size
            
    weights = list(exist_chunks_on_node.items())
    weights.sort(key=lambda item: item[1], reverse=True)
    max_weight = weights[0][1]
    nodes_with_max_weight = [tpl for tpl in weights if tpl[1]==max_weight]
    random_sel_node = random.choice(nodes_with_max_weight)[0]
    weights = [(tpl[0], tpl[0]==random_sel_node) for tpl in weights]
    return weights
    
    
# Job and Data placement strategy for (1 job/ 1 worker) use case
def base_placement(resources, job, ssd: bool = True, worker_idx=None, job_idx: int = 0, scheduler=None, data_place_alg=None):
    storage_dev = 'ssd' if ssd else 'hdd'

    # although we have sorted the nodes list, it doesn't guarantee the job can be deployed
    if scheduler == 'ff':
        node_weights = ff_nodes_weights(resources, job, job_idx)
    elif scheduler == "bf":
        node_weights = bf_nodes_weights(resources, job, job_idx)
    elif scheduler == "wf":
        node_weights = wf_nodes_weights(resources, job, job_idx)
    elif scheduler == 'csa':
        node_weights = csa_nodes_weights(resources, job, job_idx)
    else:
        node_weights = sort_nodes_by_weights(resources, job, ssd, job_idx=job_idx)

    request_gpu = job['gpu']
    request_cpu = job['cpu']

    placement = None
    for node, weight in node_weights:
        if resources['gpu'][node] >= request_gpu and resources['cpu'][node] >= request_cpu:
            placement = node
            break

    if placement is None:
        # print(request_cpu, request_gpu, resources)
        return 1e9  # computation resource is not enough for deploying the job

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

            # all_nodes = [item[0] for item in node_weights]
            # if data_place_alg != 'random':
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
            
            # else:
            #     # TODO: selected node might can't hold the chunk
            #     sel_node = all_nodes[i % len(all_nodes)]
            #     if ssd:
            #         chunk['Location'] = sel_node
            #     else:
            #         chunk['SourceLocation'] = sel_node

        if worker_idx is not None:
            worker_name = f"{job['dltdeploy_id']}-{job['job_id']}-{worker_idx}"
            if 'Jobs' not in chunk:
                chunk['Jobs'] = []
            chunk['Jobs'].append(worker_name)

    if resource_gap == 0 and placement is not None:
        job['location'] = [placement]
        resources['gpu'][placement] -= request_gpu
        resources['cpu'][placement] -= request_cpu

    return resource_gap


# Data placement strategy for (1 job / N worker) use case
def multi_worker_placement(resources, job, ssd: bool = True, job_idx: int = 0, scheduler=None, data_place_alg=None):
    resource_gap = 0
    num_workers = job['numWorkers']
    chunks_group = np.array_split(job['chunks'], num_workers)
    workers_placement = []
    job['numWorkers'] = 1
    processed_chunks = []
    job_cpy = deepcopy(job)
    for i in range(num_workers):
        job['chunks'] = chunks_group[i].tolist()
        gap = base_placement(resources, job, ssd, worker_idx=i, job_idx=job_idx, scheduler=scheduler, data_place_alg=data_place_alg)
        resource_gap += gap
        if gap == 0:
            processed_chunks.extend(job['chunks'])
            workers_placement.extend(job['location'])
        elif gap == 1e9: # caused by gpu, cpu
            break
        else: # caused by storage
            workers_placement.extend(job['location'])
            break
    
    # revoke resources assigned to previous deployable workers
    if resource_gap > 0:
        for worker in workers_placement:
            # print(f"revoke {job['gpu']} gpu {job['cpu']} cpu from worker {worker}, gap: {resource_gap}")
            resources['cpu'][worker] += job['cpu']
            resources['gpu'][worker] += job['gpu']
            for chunk in job['chunks']:
                chunk['ExistOnHDD'] = False
                chunk['ExistOnSSD'] = False
        
        for chunk in processed_chunks:
            # print(chunk['Location'])
            if chunk['Location'] is not None:
                resources['ssd'][chunk['Location']] += chunk['ChunkSize']

        job = job_cpy
        return resource_gap

    job['numWorkers'] = num_workers
    job['chunks'] = [item for sublist in chunks_group for item in sublist]
    job['location'] = workers_placement
    return resource_gap


def multi_job_placement(resources, log, running_jobs, ssd: bool = True, scheduler=None, data_place_alg=None, sort_jobs=False):
    jobs, submit_time = log['jobs'], log['submit_time']
    submit_time = datetime.strptime(submit_time, "%Y-%m-%d %H:%M:%S")
    if sort_jobs or scheduler == 'ours':
        jobs.sort(key=lambda job: job['numWorkers'])

    chunks_placement = None
    
    i = 0
    while i < len(jobs):
        
        # collect resources from finished jobs
        running_jobs.sort(key=lambda hist_job: datetime.strptime(hist_job['end_time'], "%Y-%m-%d %H:%M:%S"))
        while len(running_jobs) > 0:
            if submit_time >= datetime.strptime(running_jobs[0]['end_time'], "%Y-%m-%d %H:%M:%S"):
                hist_job = running_jobs.pop(0)
                assert len(hist_job['location']) == hist_job['numWorkers']
                for j in range(hist_job['numWorkers']):
                    loc = hist_job['location'][j]
                    cluster['cpu'][loc] += hist_job['cpu']
                    cluster['gpu'][loc] += hist_job['gpu']

                if hist_job['last']:
                    for chunk in hist_job['chunks']:
                        cluster['ssd'][chunk['Location']] += chunk['ChunkSize']
                    results.append(hist_job)
            else:
                break
                
        job = jobs[i]
        if job['deployed']:
            submit_time = submit_time + timedelta(seconds=10)
            i += 1
            continue
        
        if i > 0:
            job['chunks'] = deepcopy(chunks_placement)

        resource_gap = multi_worker_placement(resources, job, ssd, job_idx=i, scheduler=scheduler, data_place_alg=data_place_alg)    
        if resource_gap > 0:
            # if len(running_jobs) == 0:
            #     print(cluster)
            #     total_req_gpu = sum([job['gpu'] for job in log['jobs']])
            #     print('req gpu', total_req_gpu)
            #     # print(log)
            #     # print(job)
            #     exit(0)
            assert len(running_jobs) > 0
            submit_time = submit_time + timedelta(seconds=10)
            continue

        if i == 0 and resource_gap == 0:
            for chunk in job['chunks']:
                assert len(job['location']) > 0
                assert chunk['Location'] is not None
                chunk['ExistOnHDD'] = True
                chunk['ExistOnSSD'] = True
            chunks_placement = deepcopy(job['chunks'])
        job['deployed'] = True
        i += 1
        job['last'] = i == len(jobs)
        running_jobs.append(job)
        
    return True


def generate_random_ip():
    octets = [random.randint(0, 255) for _ in range(4)]
    ip_address = '.'.join(map(str, octets))
    return ip_address


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sch", type=str, choices=['ours', 'ff', 'bf', 'wf', 'csa'], default='ours')
    parser.add_argument("--cluster", type=str, choices=['venus', 'earth', 'saturn', 'uranus'], default='venus')
    parser.add_argument("--sj", action='store_true', default=False)
    args = parser.parse_args()
    
    base_dir = f'./experiments/{args.cluster}/' 
    with open(f'{base_dir}/{args.cluster}.json', 'r') as f:
        trace = json.load(f)
    trace.sort(key=lambda x: x['submit_time'])
    datasets = {}
    cluster = {}
    results = []
    
    total_nodes = {'venus': 133, 'uranus': 264, 'earth': 143, 'saturn': 262}
    num_cpus = {'venus': 48, 'uranus': 64, 'earth': 48, 'saturn': 64}
    nodes = [generate_random_ip() for _ in range(total_nodes[args.cluster])]
    for res, s in [('gpu', 8), ('cpu', num_cpus[args.cluster]), ('ssd', 2000)]:
        cluster[res] = {}
        for node in nodes:
            cluster[res][node] = s

    running_jobs = []

    t = 0
    print(args.cluster, len(trace))
    for log in trace:
        for job in log['jobs']:
            job['dltdeploy_id'] = log['deploy_id']
            job['chunks'] = log['datasource']['chunks']
            job['deployed'] = False

        # schedule the job
        multi_job_placement(cluster, log, running_jobs, scheduler=args.sch, sort_jobs=args.sj)
        
        if t % 500 == 0:
            print(args.cluster, t)
        t += 1
    
    s = '-s' if args.sj else ''
    with open(f'{base_dir}/{args.sch}{s}.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(cluster)
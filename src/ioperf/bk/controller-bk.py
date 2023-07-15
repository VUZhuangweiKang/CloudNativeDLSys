import numpy as np
import csv
from kubernetes import client, config
import zmq
import yaml
import time
import multiprocessing as mp
import subprocess
import sys
import os
import re

sys.path.insert(1, '../')
from logger import get_logger

logger = get_logger(__name__)

ns = 'default'

io_modes = [0, 1]
bandwidthes = 1000 * np.arange(1, 19, 2) # 1Gbps - 17Gbps
total_sizes = (1024 ** 3) * np.array([2**i for i in range(0, 4)]) # 1GB - 8GB
size_b = 128 * 1024
block_sizes = [] # 128KB - 1GB
while size_b <= 1024 * 1024 * 1024:
    block_sizes.append(size_b)
    size_b *= 2


config.load_kube_config()
api = client.CoreV1Api()
api_client = client.ApiClient()
network_api = client.NetworkingV1Api(api_client)
custom_api = client.CustomObjectsApi()
nodes = api.list_node().items


def get_workers():
    worker_nodes = [node for node in nodes if 'node-role.kubernetes.io/master' not in node.metadata.labels]
    worker_nodes_name = [node.metadata.name for node in worker_nodes]
    worker_addresses = [node.status.addresses[0].address for node in worker_nodes]
    return worker_nodes_name, worker_addresses


def get_master():
    for node in nodes:
        if "node-role.kubernetes.io/master" in node.metadata.labels:
            return node.metadata.name, node.status.addresses[0].address
    return None, None


def convert_bw(bw_value, bw_unit):
    """将带宽值转换为字节每秒（B/s）"""
    bw_unit = bw_unit.lower()

    if bw_unit == 'b/s':
        return bw_value
    elif bw_unit == 'kib/s':
        return bw_value * 1024
    elif bw_unit == 'mib/s':
        return bw_value * 1024**2
    elif bw_unit == 'gib/s':
        return bw_value * 1024**3
    elif bw_unit == 'tib/s':
        return bw_value * 1024**4
    else:
        raise ValueError(f"Unknown bandwidth unit: {bw_unit}")
    
    
# 从fio输出中提取带宽利用率、磁盘利用率和IOPS信息的函数
def parse_fio_output(output):
    # 匹配 "Disk stats" 部分的 "util"
    disk_util_pattern = r"util=(\d+\.\d+)%"
    disk_util_match = re.search(disk_util_pattern, output)
    disk_util = float(disk_util_match.group(1)) if disk_util_match else None

    # 匹配 "read" 和 "write" 部分的 "IOPS" 
    iops_pattern = r"IOPS=(\d+)"
    iops_matches = re.findall(iops_pattern, output, re.IGNORECASE)
    read_iops = int(iops_matches[0]) if iops_matches else None
    write_iops = int(iops_matches[1]) if len(iops_matches) > 1 else None

    # 匹配 "read" 和 "write" 部分的 "BW"
    bw_pattern = r"BW=(\d+)([a-zA-Z/]+)"
    bw_matches = re.findall(bw_pattern, output, re.IGNORECASE)

    read_bw = convert_bw(int(bw_matches[0][0]), bw_matches[0][1]) if bw_matches else None
    write_bw = convert_bw(int(bw_matches[1][0]), bw_matches[1][1]) if len(bw_matches) > 1 else None

    return disk_util, read_iops, write_iops, read_bw, write_bw


def worker_proc(pod_addr: str, worker_queue: mp.Queue, data_queue: mp.Queue):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{pod_addr}:5555")
    while True:
        mode, bandwidth, total_size, block_size = worker_queue.get()
        logger.info(f"send: {mode} {bandwidth} {total_size} {block_size}")
        socket.send_string(f"{mode} {total_size} {block_size}")
        latency = socket.recv_string()
        latency = float(latency)
        rlt = [mode, bandwidth, total_size, block_size, latency]
        data_queue.put(rlt)
        columns = ['io_mode', 'bandwidth', 'total_size', 'block_size', 'latency']
        output = {columns[i]: rlt[i] for i in range(len(columns))}
        logger.info(f"{pod_addr}: {output}")


def create_pod(node_name, nfs_svr, bandwidth):
    name = f"ioperf-on-{node_name}"
    pod_ip = None
    try:
        pod_status = api.read_namespaced_pod_status(name=name, namespace=ns)
        pod_ip = pod_status.status.pod_ip
    except client.exceptions.ApiException as e:
        if e.status == 404:
            with open('pod.yaml', 'r') as f:
                pod_body = yaml.safe_load(f)
            pod_body['metadata']['name'] = name
            pod_body['metadata']['labels']['app'] = name
            pod_body['metadata']['annotations']["kubernetes.io/ingress-bandwidth"] = f"{bandwidth}M"
            pod_body['metadata']['annotations']["kubernetes.io/egress-bandwidth"] = f"{bandwidth}M"
       
            pod_body['spec']['nodeSelector']['kubernetes.io/hostname'] = node_name
            pod_body['spec']['hostName'] = name
            pod_body['spec']['containers'][0]['env'][0]['value'] = master_addr
            pod_body['spec']['containers'][0]['env'][1]['value'] = node_name
            pod_body['spec']['containers'][0]['command'] = ["python3", "worker.py"]

            pod_body['spec']['volumes'][0]['nfs']['server'] = nfs_svr
            api.create_namespaced_pod(body=pod_body, namespace=ns)
            pod_status = api.read_namespaced_pod(name=name, namespace=ns)
            while pod_status.status.phase != 'Running':
                time.sleep(1)
                pod_status = api.read_namespaced_pod(name=name, namespace=ns)
                pod_ip = pod_status.status.pod_ip
        else:
            print(f"Error: {e}")
    return name, pod_ip


# This is unused, We can just use Pod annotation to limit Pod-Pod bandwidth when using Cilium.
# To limit the bandwidth between Pod and NFS server, we have to use TC to limit the bandwidth
# of NFS server's host machine because Pod and NFS server are not in the same network namespace.
def create_netpolicy(pod_name, bandwidth):
    with open('network_policy.yaml', 'r') as f:
        net_policy = yaml.safe_load(f)
    name = f"nw-policy-{pod_name}"
    net_policy['metadata']['name'] = name
    net_policy['spec']['endpointSelector']['matchLabels']['app'] = pod_name
    net_policy['spec']['ingressBandwidth']['rate'] = f"{bandwidth}mbit"
    net_policy['spec']['egressBandwidth']['rate'] = f"{bandwidth}mbit"
    print(net_policy)
    group='cilium.io'
    version='v2'
    namespace='default'
    plural='ciliumnetworkpolicies'
    while True:
        try:
            req = custom_api.get_namespaced_custom_object(group, version,namespace, plural, name)
            custom_api.delete_namespaced_custom_object(group, version,namespace, plural, name)
        except client.exceptions.ApiException as e:
            break
    custom_api.create_namespaced_custom_object(group, version,namespace, plural, net_policy)
        

def profiler(data_queue: mp.Queue, terminate: mp.Value):
    while not terminate.value:
        try:
            data = data_queue.get(block=False)
            mode, bandwidth, total_size, block_size, latency = data
            with open(f'data/ioperf-{mode}-{bandwidth}-{total_size}.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        except Exception:
            pass
        time.sleep(5)
            

def evaluate(iomode, bandwidth):
    logger.info(f"Total tests: {len(block_sizes) * len(total_sizes)}")
    data_queue = mp.Queue()
    terminate = mp.Value('i', 0)
    workers_address = []
    workers_names = []
    # create a pod on each node
    for i in range(len(cluster_workers_address)):
        nfs_svr = cluster_workers_address[(i + 1) % len(cluster_workers_address)]
        # limit bandwidth on the NFS Server
        if iomode == 1:
            command = 'ssh %s@%s "sudo ./tc.sh %d"' % (os.getlogin(), nfs_svr, bandwidth)
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
            output, error = process.communicate()
            output, error = output.decode(), error.decode()
            logger.info(str(output))
            if len(error) > 0:
                logger.error(str(error))
        pod_name, pod_ip = create_pod(cluster_workers_hostname[i],
                                      cluster_workers_address[(i + 1) % len(cluster_workers_address)],
                                      bandwidth=bandwidth)
        workers_address.append(pod_ip)
        workers_names.append(pod_name)
        logger.info(f"add pod {pod_name} {pod_ip}")
    worker_queues = []
    workers = []
    for i in range(len(cluster_workers_address)):
        worker_queue = mp.Queue()
        worker_queues.append(worker_queue)
        proc = mp.Process(target=worker_proc, args=(workers_address[i], worker_queue, data_queue), daemon=True)
        proc.start()
        workers.append(proc)

    mp.Process(target=profiler, args=(data_queue, terminate), daemon=True).start()
    
    for i, total_size in enumerate(total_sizes):
        worker_idx = i % len(workers)
        for block_size in block_sizes:
            worker_queues[worker_idx].put((iomode, bandwidth, total_size, block_size))
    
    # waiting for all workers down
    while True:
        c = 0
        for i in range(len(worker_queues)):
            if worker_queues[i].qsize() == 0:
                c += 1
        if c == len(worker_queues):
            break
    
    # inform profiler to stop
    terminate.value = 1
    
    for i in range(len(workers)):
        worker_queues[i].close()
        workers[i].terminate()

    # Delete pod
    for name in workers_names:
        try:
            api.delete_namespaced_pod(name, namespace=ns, body=client.V1DeleteOptions())

            # Wait for pod to be deleted
            while True:
                try:
                    api.read_namespaced_pod(name, namespace=ns)
                    time.sleep(1)
                except client.ApiException as e:
                    if e.status == 404:
                        break
                    else:
                        raise e
            print(f"Pod {name} deleted successfully")
        except client.ApiException as e:
            print('Exception when calling CoreV1Api->delete_namespaced_pod: %s\n' % e)


if __name__ == "__main__":
    cluster_workers_hostname, cluster_workers_address = get_workers()
    master, master_addr = get_master()
    # evaluate(iomode=0, bandwidth=max(bandwidth))
    for bw in bandwidthes:
        evaluate(iomode=1, bandwidth=bw)
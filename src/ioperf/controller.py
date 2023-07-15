import numpy as np
from kubernetes import client, config
import zmq
import yaml
import time
import multiprocessing as mp
import subprocess
import sys
import os
import pandas as pd

sys.path.insert(1, '../')
from logger import get_logger

logger = get_logger(__name__)

ns = 'default'

io_modes = [0, 1]
# bandwidthes = 1000 * np.arange(1, 19, 2) # 1Gbps - 17Gbps
bandwidthes = 1000 * np.arange(1, 11, 1) # 1Gbps - 10Gbps
# total_sizes = (1024 ** 3) * np.array([2**i for i in range(0, 3)]) # 1GB - 4GB
total_sizes = [4*1024*1024*1024]
size_b = 128 * 1024
block_sizes = [] # 128KB - 1GB
while size_b <= 1024 * 1024 * 1024:
    block_sizes.append(int(size_b))
    size_b *= 1.2


config.load_kube_config()
api = client.CoreV1Api()
api_client = client.ApiClient()
network_api = client.NetworkingV1Api(api_client)
custom_api = client.CustomObjectsApi()
nodes = api.list_node().items


def get_workers():
    worker_nodes = [node for node in nodes if 'node-role.kubernetes.io/control-plane' not in node.metadata.labels]
    worker_nodes_name = [node.metadata.name for node in worker_nodes]
    worker_addresses = [node.status.addresses[0].address for node in worker_nodes]
    return worker_nodes_name, worker_addresses


def get_master():
    for node in nodes:
        if "node-role.kubernetes.io/control-plane" in node.metadata.labels:
            return node.metadata.name, node.status.addresses[0].address
    return None, None


def create_pod(node_name, nfs_svr):
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
            # pod_body['metadata']['annotations']["kubernetes.io/ingress-bandwidth"] = f"{bandwidth}M"
            # pod_body['metadata']['annotations']["kubernetes.io/egress-bandwidth"] = f"{bandwidth}M"
       
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
            

def worker(node, nfs_svr, iomode, bandwidth, total_size):
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

    # create pod on worker node
    pod_name, pod_ip = create_pod(node, nfs_svr)
    logger.info(f"add pod {pod_name} {pod_ip}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{pod_ip}:5555")
    
    results = []
    for block_size in block_sizes:
        logger.info(f"send: {iomode} {bandwidth} {total_size} {block_size}")
        socket.send_string(f"{iomode} {total_size} {block_size}")
        latency = socket.recv_string()
        latency = float(latency)
        rlt = [iomode, bandwidth, total_size, block_size, latency]
        results.append(rlt)
        columns = ['io_mode', 'bandwidth', 'total_size', 'block_size', 'latency']
        output = {columns[i]: rlt[i] for i in range(len(columns))}
        logger.info(f"{pod_ip}: {output}")

    # Delete pod
    try:
        api.delete_namespaced_pod(pod_name, namespace=ns, body=client.V1DeleteOptions())

        # Wait for pod to be deleted
        while True:
            try:
                api.read_namespaced_pod(pod_name, namespace=ns)
                time.sleep(1)
            except client.ApiException as e:
                if e.status == 404:
                    break
                else:
                    raise e
        print(f"Pod {pod_name} deleted successfully")
    except client.ApiException as e:
        print('Exception when calling CoreV1Api->delete_namespaced_pod: %s\n' % e)
    
    return results
    
def evaluate(node, nfs_svr, myschedules):
    for schedule in myschedules:
        iomode, bandwidth, total_size = schedule
        rlt = worker(node, nfs_svr, iomode, bandwidth, total_size)
        rlt = pd.DataFrame(rlt, columns=['iomode', 'bandwidth', 'total_size', 'block_size', 'latency'])
        rlt.to_csv(f"data/ioperf-{iomode}-{bandwidth}-{total_size}.csv", index=None)


if __name__ == "__main__":
    cluster_workers_hostname, cluster_workers_address = get_workers()
    master, master_addr = get_master()
    for iomode in io_modes:
        schedules = []
        if iomode == 0:
            bws = bandwidthes[-1:]
        elif iomode == 1:
            bws = bandwidthes
        else:
            continue
        for bw in bws:
            for total_size in total_sizes:
                schedules.append([iomode, bw, total_size])
        np.random.shuffle(schedules)
        schedules = np.array_split(schedules, len(cluster_workers_address))
        eval_procs = []
        for i, schedule in enumerate(schedules):
            proc = mp.Process(target=evaluate, args=(cluster_workers_hostname[i],
                                                    cluster_workers_address[(i+1)%len(cluster_workers_address)],
                                                    schedule), daemon=True)
            eval_procs.append(proc)
            proc.start()
        for proc in eval_procs:
            proc.join()
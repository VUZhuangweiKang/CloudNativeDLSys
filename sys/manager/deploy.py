import yaml
from kubernetes import client, config
import socket
import sys
import argparse


with open("deploy_template.yaml") as file:
    dep_manifest = yaml.safe_load(file)


config.load_kube_config()
core_api = client.CoreV1Api()
ret = core_api.list_node(pretty=True)
nodes = []
for node in ret.items:
    for condition in node.status.conditions:
        if condition.type == "Ready" and condition.status == "True":
            for address in node.status.addresses:
                if address.type == "InternalIP":
                    nodes.append((node.metadata.name, address.address))

volumes = []
volume_mounts = []
for device in ['hdd', 'ssd']:
    for node_name, node_ip in nodes:
        volumes.append({
            'name': f"{node_name}-{device}",
            "nfs": {
                "server": node_ip,
                "path": f"/nfs/{device}"
            }
        })
        if node_name == socket.gethostname():
            volume_mounts.append({
                "name": f"{node_name}-{device}",
                "mountPath": f"/mnt/nfs/{device}/local"
            })
        else:
            volume_mounts.append({
                "name": f"{node_name}-{device}",
                "mountPath": f"/mnt/nfs/{device}/{node_ip}"
            })
dep_manifest['spec']['template']['spec']['volumes'].extend(volumes)
for i in range(len(dep_manifest['spec']['template']['spec']['containers'])):
    dep_manifest['spec']['template']['spec']['containers'][i]['volumeMounts'].extend(volume_mounts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merging', action="store_true", default=False, help='Merging small files into data blocks')
    parser.add_argument('--block_size', type=int, default=0, help='fixed block size in bytes')
    parser.add_argument('--threadpool_size', type=int, default=0, help='fixed thread pool size in DLCJob')
    parser.add_argument('--schedule_alg', choices=['ff', 'bf', 'dlsys'], default='dlsys', help='Scheduling algorithm')
    parser.add_argument('--data_place_alg', choices=['random', 'local'], default='local', help='Data placement algorithm')
    
    args = parser.parse_args()
    
    if args.merging:
        dep_manifest['spec']['template']['spec']['containers'][0]['args'] = [f"python3 ManagerService.py --merging --block_size {args.block_size} --threadpool_size {args.threadpool_size} --schedule_alg {args.schedule_alg}"]
    else:
        dep_manifest['spec']['template']['spec']['containers'][0]['args'] = [f"python3 ManagerService.py --threadpool_size {args.threadpool_size} --schedule_alg {args.schedule_alg} --data_place_alg {args.data_place_alg}"]
    
    # dep_manifest['spec']['template']['spec']['containers'][0]['args'] = ["bash"]


    with client.ApiClient() as api_client:
        v1 = client.AppsV1Api(api_client)
        dep = v1.create_namespaced_deployment(
            namespace="default",
            body=dep_manifest,
        )
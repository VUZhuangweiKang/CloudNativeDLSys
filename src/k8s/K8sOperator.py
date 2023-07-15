from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from typing import Dict, List
from commons import utils
import time
from collections import defaultdict


def get_node_ip(node):
    for address in node.status.addresses:
        if address.type == 'InternalIP':
            node_ip = address.address
            return node_ip

def convert_to_bytes(size_str):
    """
    Convert a string size to bytes. The size_str is expected to be a number
    followed by a unit: Gi, G, GB, Mi, M, MB. The function returns the size
    in bytes.
    """
    units = {
        'G': 1 << 30,
        'GB': 1 << 30,
        'Gi': 1 << 30,
        'M': 1 << 20,
        'MB': 1 << 20,
        'Mi': 1 << 20,
        'K': 1 << 10,
        'KB': 1 << 10,
        'Ki': 1 << 10
    }
    
    # Default return value
    num_bytes = 0

    # Separate number from unit
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            num_bytes = float(size_str[:-len(unit)]) * multiplier
            break

    return int(num_bytes)


def convert_cpu_to_float(cpu_str):
    if cpu_str.endswith('m'):
        return float(cpu_str.rstrip('m')) / 1000
    else:
        return float(cpu_str)
    
    
class K8sOperator:
    def __init__(self):
        config.load_incluster_config()
        self.client = client.CoreV1Api()
        self.api_instance = client.CustomObjectsApi()
        self.watch = watch.Watch()

    def list_nodes_ip(self):
        nodes_ip = []
        for node in self.client.list_node().items():
            nodes_ip.append(get_node_ip(node))
        return nodes_ip

    @staticmethod
    def filter_worker_nodes(nodes):
        worker_nodes = []
        for node in nodes:
            if 'node-role.kubernetes.io/master' not in node.metadata.labels and 'node-role.kubernetes.io/control-plane' not in node.metadata.labels:
                worker_nodes.append(node)
        return worker_nodes
    
    def get_free_gpus(self) -> Dict[str, float]:
        nodes = self.client.list_node().items
        worker_nodes = self.filter_worker_nodes(nodes)
        free_gpus = defaultdict(float)
        for node in worker_nodes:
            res = dict(node.status.allocatable)
            # res output example: {'cpu': '40', 'ephemeral-storage': '824807653649', 'hugepages-1Gi': '0', 'hugepages-2Mi': '0', 'memory': '65634784Ki', 'pods': '110'}
            try:
                free_gpus[get_node_ip(node)] = int(res["nvidia.com/gpu"])
            except:
                # free_gpus[get_node_ip(node)] = int(res['cpu'])
                
                # for exp only
                free_gpus[get_node_ip(node)] = int(res['cpu'])
        return free_gpus

    def get_free_storage(self, base_dir: str) -> Dict[str, float]:
        nodes = self.client.list_node().items
        worker_nodes = self.filter_worker_nodes(nodes)
        free_space = defaultdict(float)
        for node in worker_nodes:
            node_ip = get_node_ip(node)
            path = f"{base_dir}/{node_ip}"
            free_space[node_ip] = utils.get_free_space(path)
        return free_space

    def get_free_resources(self, hdd_dir, ssd_dir) -> Dict[str, float]:
        nodes = self.client.list_node().items
        worker_nodes = self.filter_worker_nodes(nodes)
        resources = defaultdict(lambda: defaultdict(float))
        
        for node in worker_nodes:
            node_ip = get_node_ip(node)
            node_name = node.metadata.name
            
            # res output example: {'cpu': '40', 'ephemeral-storage': '824807653649', 'hugepages-1Gi': '0', 'hugepages-2Mi': '0', 'memory': '65634784Ki', 'pods': '110'}
            alloc_res = dict(node.status.allocatable)
            # for exp only
            alloc_res['cpu'] = int(alloc_res['cpu'])
            
            field_selector = f'spec.nodeName={node_name},status.phase!=Succeeded,status.phase!=Failed'
            pods_on_node = self.client.list_pod_for_all_namespaces(field_selector=field_selector)

            used_cpu = 0
            used_memory = 0
            used_gpu = 0
            used_ephemeral_storage = 0

            for pod in pods_on_node.items:
                for container in pod.spec.containers:
                    requests = container.resources.requests if container.resources else None
                    if requests:
                        if "cpu" in requests:
                            used_cpu += convert_cpu_to_float(requests["cpu"])
                        if "memory" in requests:
                            used_memory += convert_to_bytes(requests["memory"])
                        if 'nvidia.com/gpu' in requests:
                            used_gpu += requests['nvidia.com/gpu']
                        if 'ephemeral-storage' in requests:
                            used_ephemeral_storage += convert_to_bytes(requests['ephemeral-storage'])
                            
            resources['cpu'][node_ip] = int(alloc_res['cpu']) - used_cpu
            if "nvidia.com/gpu" in alloc_res:
                resources['gpu'][node_ip] = int(alloc_res["nvidia.com/gpu"]) - used_gpu
            else:
                resources['gpu'][node_ip] = resources['cpu'][node_ip]
            resources['ephemeral-storage'][node_ip] = convert_to_bytes(alloc_res["ephemeral-storage"]) - used_ephemeral_storage
            resources['hdd'][node_ip] = int(utils.get_free_space(f"{hdd_dir}/{node_ip}"))
            resources['ssd'][node_ip] = int(utils.get_free_space(f"{ssd_dir}/{node_ip}"))
            resources['memory'][node_ip] = convert_to_bytes(alloc_res['memory']) - used_memory
        return resources


class DLTDeployOperator(K8sOperator):
    namespace = "default"
    group = "docgroup.com"
    version = "v1alpha1"
    plural = "dltdeployments"

    def __init__(self):
        super().__init__()

    def deploy(self, jobs) -> bool:
        cred = jobs[0].cred
        uid = jobs[0].dltdeploy_id
        body = {
            "apiVersion": f"{self.group}/{self.version}",
            "kind": "DLTDeployment",
            "metadata": {
                "name": uid,
                "namespace": f"{self.namespace}"
            },
            "spec": {
                "credential": cred,
                "jobs": [{"name": job.name, "nodes": job.nodes, "uid": job.job_id, **job.spec} for job in jobs]
            }
        }
        try:
            self.api_instance.create_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                body=body
            )
            return True
        except ApiException as e:
            print(e)
            return False

    def get_dltdeployment(self, uid: str):
        try:
            dltdeployment = self.api_instance.get_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                name=uid,
            )
            return dltdeployment
        except ApiException as e:
            return None
        
    def is_dltdeployment_exists(self, uid: str) -> bool:
        return self.get_dltdeployment(uid) is not None

    def delete_dltdeployment(self, uid: str) -> bool:
        try:
            self.api_instance.delete_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                name=uid,
                grace_period_seconds=0
            )
            if self.is_dltdeployment_exists(uid):
                # Watch the status of the pod
                for event in self.watch.stream(self.api_instance.list_namespaced_custom_object, namespace='default'):
                    # Break the loop and return True if the pod is not found within the time limit
                    if event['object'].metadata.name == uid:
                        if event['type'] == 'DELETED':
                            return True
                
                # If pod not deleted after timeout, return False
                return False
            else:
                return True
        except ApiException as e:
            # Return False if the pod was not found (which means it was already deleted)
            if e.status == 404:
                return True
            else:
                raise
    
    def is_pod_exists(self, pod_name: str) -> bool:
        try:
            self.client.read_namespaced_pod(name=pod_name, namespace='default')
            return True
        except ApiException as e:
            if e.status == 404:
                # Pod doesn't exist
                return False
            raise
    
    def is_pod_running(self, pod_name: str) -> bool:
        try:
            response = self.client.read_namespaced_pod(name=pod_name, namespace='default')
            if response.status.phase != 'Running':
                return False
            return True
        except ApiException as e:
            if e.status == 404:
                # Pod doesn't exist
                return False
            raise
    
    def is_dltworker_running(self, dltdeploy: str, job_id=str) -> bool:
        deploy = self.api_instance.get_namespaced_custom_object(
            group=self.group,
            version=self.version,
            namespace=self.namespace,
            plural=self.plural,
            name=dltdeploy,
        )
        running = True
        for job_info in deploy['spec']['jobs']:
            if 'workers' in job_info:
                for worker in job_info['workers']:
                    pod_info = self.client.read_namespaced_pod(namespace=self.namespace, name=worker)
                    for container_status in pod_info.status.container_statuses:
                        if job_id in container_status.name:
                            running = running and container_status.state.running is not None
        return running
        
    def delete_pod(self, pod_name: str) -> bool:
        try:
            # Send delete pod request
            self.client.delete_namespaced_pod(name=pod_name, namespace='default', grace_period_seconds=0)
            
            # Watch the status of the pod
            if self.is_pod_exists(pod_name):
                for event in self.watch.stream(self.client.list_namespaced_pod, namespace='default'):
                    # Break the loop and return True if the pod is not found within the time limit
                    if event['object'].metadata.name == pod_name:
                        if event['type'] == 'DELETED':
                            return True
            else:
                return True
                
            # If pod not deleted after timeout, return False
            return False

        except ApiException as e:
            # Return False if the pod was not found (which means it was already deleted)
            if e.status == 404:
                return True
            else:
                raise
    
    def delete_config_map(self, name):
        namespace = 'default'
        
        # Delete the ConfigMap
        try:
            self.client.delete_namespaced_config_map(name, namespace, grace_period_seconds=0)
        except ApiException as e:
            if e.status != 404:
                print("Failed to delete ConfigMap due to: %s" % e)
                return

        # Poll until the ConfigMap no longer exists.
        while True:
            try:
                self.client.read_namespaced_config_map(name, namespace)
                print('Waiting for ConfigMap to be deleted')
                time.sleep(1)
            except ApiException as e:
                if e.status == 404:
                    print('ConfigMap is deleted')
                    break
                else:
                    print("Failed to get ConfigMap status due to: %s" % e)
                    return

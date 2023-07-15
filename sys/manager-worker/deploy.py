import yaml
from kubernetes import client, config


with open("./daemonset_template.yaml") as file:
    dmn_manifest = yaml.safe_load(file)


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
        volume_mounts.append({
            "name": f"{node_name}-{device}",
            "mountPath": f"/mnt/nfs/{device}/{node_ip}"
        })
dmn_manifest['spec']['template']['spec']['volumes'].extend(volumes)
for i in range(len(dmn_manifest['spec']['template']['spec']['containers'])):
    dmn_manifest['spec']['template']['spec']['containers'][i]['volumeMounts'].extend(volume_mounts)

with client.ApiClient() as api_client:
    v1 = client.AppsV1Api(api_client)
    dep = v1.create_namespaced_daemon_set(
        namespace="default",
        body=dmn_manifest,
    )
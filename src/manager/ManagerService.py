import json
import multiprocessing as mp
import zmq
import sys
sys.path.append("../")
from collections import defaultdict

from configurations import ConfigParser
from manager import *
from manager.ChunkOps import load_chunks, download_chunks, chunk_extractor
from manager.Manager import scheduler, executer
import argparse


def save_job_info(job: JobQueueItem):
    dataset_etags = defaultdict(list)
    for chunk in job.chunks:
        dataset_etags[chunk['Category']].append(chunk['ETag'])
    mongo_operator.add_job(job.cred['username'], job.job_id, job.dltdeploy_id, job.spec['numWorkers'], job.spec['datasource'], job.spec['resource_requests'], dict(dataset_etags))


class ManagerService:
    def __init__(self) -> None:
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{dbus_configs['manager']['port']}")
        
        while True:
            command, message = self.socket.recv_multipart()
            command = command.decode()
            if command == 'deploy':
                req = pb.DeployRequest.FromString(message)
                resp = self.deploy(req)
            elif command == 'join':
                req = pb.WorkerJoinRequest.FromString(message)
                resp = self.join(req)
            elif command == 'handle_datamiss':
                req = pb.DataMissRequest.FromString(message)
                resp = self.handle_datamiss(req)
            elif command == 'heartbeat':
                self.socket.send(b"")
                continue
            elif command == 'del_dltpod':
                self.socket.send(b"")
                req = pb.DelDLTPodRequest.FromString(message)
                # self.delete_dltpod(req)
                continue
            else:
                continue
            self.socket.send(resp.SerializeToString())
    
    def deploy(self, request: pb.DeployRequest) -> pb.DeployResponse:
        cred = request.credential
        jobs = json.loads(request.jobs)
        cred = utils.protobuf_msg_to_dict(cred)
        resp = pb.DeployResponse()
        try:
            dltdeploy_id = f"{request.name}-{utils.random_alphabetic(5)}"
            # generate all jobs IDs
            job_ids = [f"{job['name']}-{utils.random_alphabetic(5)}".lower() for job in jobs]  # Job ID, will be the uid value of each job
            tmp = {}
            for i, job in enumerate(jobs):
                job_id = job_ids[i]
                chunks = load_chunks(cred, job['datasource'])
                
                # we don't allow user to specify thre resource requests per worker
                # request = limit
                job['resource_requests'] = {
                    "GPU": 8,
                    "CPU": 8,
                    "Memory": int(8e9)
                }
                
                job_obj = JobQueueItem(name=job['name'], job_id=job_id, dltdeploy_id=dltdeploy_id, cred=cred, spec=job, chunks=chunks, peers=job_ids)
                save_job_info(job_obj)
                tmp[job_id] = job_obj
            job_queue.update(tmp)
            resp.rc = pb.RC.DEPLOYED
        except:
            resp.rc = pb.RC.FAILED
            resp.resp = "failed to deploy the job"
        return resp

    def join(self, request: pb.WorkerJoinRequest) -> pb.WorkerJoinResponse:
        node, worker = request.node_ip, request.worker_ip
        resp = pb.WorkerJoinResponse()
        try:
            workers_url[node] = f"{worker}:{dbus_configs['worker']['port']}"
            logger.info(f"worker {worker} from node {node} attempted to join")
            resp.rc = True
        except Exception:
            resp.rc = False
        return resp

    def handle_datamiss(self, request: pb.DataMissRequest) -> pb.DataMissResponse:
        cred = request.cred
        rc = self.auth_client(cred, conn_check=True)
        if rc != pb.RC.CONNECTED:
            return
        print('DataMissService Log: ', request.etag)
        # download data
        chunk = mongo_operator.find_etag(request.etag)
        s3_operator = S3Operator(mongo_operator.get_s3auth(**utils.protobuf_msg_to_dict(cred)))
        chunk = download_chunks(chunks=[chunk], s3_operator=s3_operator, bucket_name=chunk['Bucket'])
        chunk_extractor(chunks=[chunk])
        resp = pb.DataMissResponse()
        resp.response = True
        return resp
    
    def delete_dltpod(self, request: pb.DelDLTPodRequest):
        dltpod = request.dltpod
        dltdeploy = '-'.join(dltpod.split('-')[:-3])
        k8s_operator.delete_pod(dltpod)
        count = 0
        pod_list = k8s_operator.client.list_namespaced_pod("default")
        for pod in pod_list.items:
            if dltdeploy in pod.metadata.name and pod.status.phase == "Running":
                count += 1
        if count == 0:
            if not k8s_operator.delete_dltdeployment(dltdeploy):
                logger.error(f"failed to delete DLTDeployment {dltdeploy}")
            else:
                logger.info(f"delete DLTDeployment {dltdeploy}.")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merging', action="store_true", default=False, help='Merging small files into data blocks')
    parser.add_argument('--block_size', type=int, default=0, help='fixed block size in bytes')
    parser.add_argument('--threadpool_size', type=int, default=0, help='fixed thread pool size in DLCJob')
    parser.add_argument('--schedule_alg', choices=['ff', 'bf', 'dlsys'], default='dlsys', help='Job placement algorithm')
    parser.add_argument('--data_place_alg', choices=['random', 'local'], default='local', help='Data placement algorithm')
    args = parser.parse_args()
    
    context = zmq.Context()
    dbus_configs = ConfigParser(['dbus']).get('dbus')
    mp_manager = mp.Manager()
    workers_url = mp_manager.dict()
    job_queue = mp_manager.dict()
    deployable_job_queue = mp.Queue()

    scheduler_process = mp.Process(target=scheduler, args=(workers_url, job_queue, deployable_job_queue, args.schedule_alg, args.data_place_alg), daemon=True)
    executor_process = mp.Process(target=executer, args=(workers_url, deployable_job_queue, args.merging, args.block_size, args.threadpool_size), daemon=True)
    scheduler_process.start()
    executor_process.start()
    ManagerService()
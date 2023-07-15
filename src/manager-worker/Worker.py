import os
import shutil
import socket
import subprocess
import time
import tarfile, zipfile
import numpy as np
import pickle
# import torch
import glob
import zmq
import multiprocessing as mp
import databus.dbus_pb2 as pb
from cloudbucket.CloudBucketOperator import S3Operator
from configurations import ConfigParser
from commons import utils
from logger import get_logger
import concurrent.futures

logger = get_logger(__name__)
TIMEOUT_MILLSEC = 5000  # 5s


def get_free_space(path):
    command = "df %s | awk '{print $4}' | tail -n 1" % path
    free_space = subprocess.check_output(command, shell=True)
    return int(free_space)


def get_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip


def zcat(fpath):
    import subprocess
    file_type = fpath.split('.')[-1]
    if file_type in ['tar', 'gz', 'bz2']:
        cmd = f"zcat {fpath} | wc -c"
        s = subprocess.check_output(cmd, shell=True)
    elif file_type in ["zip"]:
        cmd = "unzip -l %s | tail -n 1 | awk '{print $1}'" % fpath
        s = subprocess.check_output(cmd, shell=True)
    else:
        s = os.path.getsize(fpath)
    return int(s)


class ManagerWorkerService:
    def __init__(self) -> None:
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{dbus_config['worker']['port']}")
        while True:
            command, message = self.socket.recv_multipart()
            command = command.decode()
            if command == 'download':
                req = pb.DownloadFileRequest.FromString(message)
                resp = self.download(req)
            elif command == 'extract':
                req = pb.ExtractFileRequest.FromString(message)
                resp = self.extract(req)
            elif command == 'move':
                req = pb.MoveChunkRequest.FromString(message)
                resp = self.move(req)
            elif command == 'merge':
                req = pb.MergeFileRequest.FromString(message)
                resp = self.merge(req)
            elif command == 'delete':
                req = pb.DelChunkRequest.FromString(message)
                resp = self.delete_chunk(req)
            elif command == 'inspect_chunk':
                req = pb.InspectChunkRequest.FromString(message)
                resp = self.inspect_chunk(req)
            else:
                continue
            
            self.socket.send(resp.SerializeToString())
            
    def download(self, request: pb.DownloadFileRequest) -> pb.DownloadFileResponse:
        s3auth, bucket, keys, destinations = request.s3auth, request.bucket, request.keys, request.destinations
        
        def helper(key, dst):
            cost = np.inf
            size = np.inf
            logger.info(f"downloading {dst} on hdd from {bucket}:{key}.")
            if os.path.exists(dst):
                # the ssd file has been extracted
                logger.info(f"{dst} exist.")
                if os.path.isdir(dst):
                    size = 0
                    for ele in os.scandir(dst):
                        size += os.path.getsize(ele)
                else:
                    size = zcat(dst)
                cost = 0  # the cost must be in database
            else:
                start = time.time()
                s3_operator = S3Operator(utils.protobuf_msg_to_dict(s3auth))
                if s3_operator.download_obj(bucket, key, dst):
                    logger.info(f"downloaded {dst} on hdd from {bucket}:{key}.")
                    cost = time.time() - start
                    size = zcat(dst)
            return size, cost
                    
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for i in range(len(keys)):
                futures.append(executor.submit(helper, keys[i], destinations[i]))
        
        resp = pb.DownloadFileResponse()
        for future in concurrent.futures.as_completed(futures):
            rlt = future.result()
            resp.sizes.append(rlt[0])
            resp.costs.append(rlt[1])
        return resp

    def extract(self, request: pb.ExtractFileRequest) -> pb.ExtractFileResponse:
        sources = request.sources
        destinations = request.destinations
        
        def helper(src, dst):
            cost = np.inf
            start = time.time()
            logger.info(f"extracting {src} to {dst}")
            if not os.path.isdir(src) and (tarfile.is_tarfile(src) or zipfile.is_zipfile(src)):
                if not os.path.exists(src):
                    os.makedirs(src)
                if not os.path.exists(dst):
                    os.mkdir(dst)
                try:
                    cmd = f"pigz -dc -p {os. cpu_count()} {src} | tar xC {dst}"
                    os.system(cmd)
                    cost = time.time() - start
                except:
                    pass
                logger.info(f"extracted {dst} on ssd.")
            else:
                cost = 0
            return cost
            
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for i in range(len(sources)):
                futures.append(executor.submit(helper, sources[i], destinations[i]))
        
        resp = pb.ExtractFileResponse()
        for future in concurrent.futures.as_completed(futures):
            resp.costs.append(future.result())
        return resp

    def inspect_chunk(self, request: pb.InspectChunkRequest) -> pb.InspectChunkResponse:
        total_size = 0
        file_count = 0
        
        for dirpath, dirnames, filenames in os.walk(request.chunk_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        resp = pb.InspectChunkResponse()
        resp.total_size = total_size
        resp.file_count = file_count
        return resp
    
    def move(self, request: pb.MoveChunkRequest) -> pb.MoveChunkResponse:
        try:
            if not os.path.exists(request.dst):
                shutil.move(src=request.src, dst=request.dst)
            logger.info(f"move file from {request.src} to {request.dst}")
            success=True
        except:
            success=False
        resp = pb.MoveChunkResponse()
        resp.success = success
        return resp
    
    def delete_chunk(self, request: pb.DelChunkRequest) -> pb.DelChunkResponse:
        delete_source = request.delete_source
        resp = pb.DelChunkResponse()
        try:
            shutil.rmtree(f"{common_config['ssd_base_dir']}/local/{request.etag}")
            if delete_source:
                shutil.rmtree(f"{common_config['hdd_base_dir']}/local/{request.etag}")
            resp.response = f"delete chunk {request.etag}"
        except Exception as ex:
            resp.response = str(ex)
        return resp
        
    def merge(self, request: pb.MergeFileRequest) -> pb.MergeFileResponse:
        dir_path = f"{common_config['ssd_base_dir']}/local/{request.etag}"
        
        files = []
        for path in glob.glob(f"{dir_path}/*"):
            if os.path.isfile(path):
                files.append(path)
        
        resp = pb.MergeFileResponse()

        if len(files) == 0:
            resp.rc = False
            resp.error = f"Directory: {dir_path} is empty."
            return resp
            
        try:
            total_files = 0
            k = request.blockSize
            
            logger.info(f"Merging {len(files)} files...")

            total_blocks = 0
            if request.numSamples == -1:
                use_num_samples = len(files)
            else:
                use_num_samples = min(request.numSamples, len(files))
            
            for start in range(0, use_num_samples, k):
                end = start + k
                block_files = files[start:end]
                if k > 1:
                    combined = []
                    for f in block_files:
                        combined.append([f, open(f, 'rb').read()])
                        os.remove(f)
                    block_len = len(combined)
                    with open(block.name, 'wb') as f:
                        pickle.dump(combined, f)
                else:
                    block_len = len(block_files)
                
                # Create the block and set its properties
                block = resp.blocks.add()
                block.name = f"{dir_path}/merged_{total_blocks}.bin"
                block.length = block_len  # should be equal to len(file_range)
                total_files += block.length                    
                total_blocks += 1 

            logger.info(f"Merged {total_files} files into {total_blocks} blocks")
            resp.rc = True
        except Exception as ex:
            resp.rc = False
            resp.error = str(ex)

        return resp        
            

def join_cluster():
    poller = zmq.Poller()
    connected = False
    def reset_socket():
        socket = context.socket(zmq.REQ)
        poller.register(socket, zmq.POLLIN)
        socket.connect(f"tcp://{dbus_config['manager']['hostname']}:{dbus_config['manager']['port']}")
        return socket

    socket = reset_socket()
    while True:
        if not connected:
            join_req = pb.WorkerJoinRequest()
            join_req.node_ip = node_ip
            join_req.worker_ip=get_ip()
            socket.send_multipart([b"join", join_req.SerializeToString()])
            logger.info("try to join the cluster.")
            socks = dict(poller.poll(TIMEOUT_MILLSEC))
            if socket in socks and socks[socket] == zmq.POLLIN:
                message = socket.recv()
                logger.info("joined the cluster.")
                join_resp = pb.WorkerJoinResponse.FromString(message)
                
                if not join_resp.rc:
                    logger.error(f"Worker on node {node_ip} failed to join the system.")
                    raise Exception
                else:
                    connected = True
            else:
                connected = False
                logger.error("Lost connection to manager, reconnecting...")
                socket.close()
                socket = reset_socket()
        else:
            socket.send_multipart([b"heartbeat", b""])
            socks = dict(poller.poll(TIMEOUT_MILLSEC))
            if socket in socks and socks[socket] == zmq.POLLIN:
                socket.recv()
                connected = True
            else:
                connected = False
                logger.error("Lost connection to manager, reconnecting...")
                socket.close()
                socket = reset_socket()

        time.sleep(TIMEOUT_MILLSEC/1000)


if __name__ == '__main__':
    context = zmq.Context()
    node_ip = os.getenv("NODE_IP")
    assert node_ip is not None
    parser = ConfigParser(components=['commons', 'dbus'])
    common_config = parser.get('commons')['data']
    dbus_config = parser.get('dbus')
    
    proc1 = mp.Process(target=join_cluster, daemon=True)
    proc1.start()
    ManagerWorkerService()
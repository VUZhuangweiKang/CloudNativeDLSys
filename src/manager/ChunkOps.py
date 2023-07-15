import bson
import zmq
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Union, Tuple
import time
import os
import shutil
import math
import concurrent
import multiprocessing as mp

from manager import utils, logger, ssd_dir, hdd_dir, mongo_operator, preparing_chunks, evict_configs
from databus import dbus_pb2 as pb
from cloudbucket.CloudBucketOperator import S3Operator
from database.MongoOperator import ChunkStatus


def load_chunks(cred: Dict[str, Union[str, Dict]], datasource: Dict[str, Dict[str, Dict[str, List]]]) -> List[Dict]:
    s3_operator = S3Operator(mongo_operator.get_s3auth(**cred))

    def filter_chunks(s3_page: List[Dict]) -> List[Dict]:
        """
        Set Exist and ETag fields for s3 returned objects
        Args:
            s3_page: S3 page contents

        Returns: List[Dict]
        """
        etags = {}
        for i in range(len(s3_page)):
            if s3_page[i]['Size'] == 0:
                continue
            s3_page[i]["ETag"] = s3_page[i]["ETag"].strip('"')  # ETag value from S3 contains " sometimes
            etags[s3_page[i]["ETag"]] = s3_page[i]["LastModified"]

        results = mongo_operator.filter_etags(list(etags.keys()))
        existing_etags = {item['ETag']: item for item in results}

        chunks_ = []
        for info in s3_page:
            if info['Size'] == 0:
                continue
            last_modified = bson.timestamp.Timestamp(int(info['LastModified'].timestamp()), inc=1)
            if info['ETag'] not in existing_etags or last_modified > existing_etags[info['ETag']]['LastModified']:
                info['ExistOnHDD'] = False
                info['ExistOnSSD'] = False
            else:
                info = existing_etags[info['ETag']]
            chunks_.append(info)
        return chunks_

    chunks = []
    bucket_name = datasource['bucket']
    for dataset_name in ['train', 'validation', 'test']:
        if dataset_name not in datasource['keys']:
            continue
        keys = datasource['keys'][dataset_name]
        subset = []
        for prefix in keys:
            contents = s3_operator.paginate(bucket_name, prefix)
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for page in contents:
                    futures.append(executor.submit(filter_chunks, page))
            for future in concurrent.futures.as_completed(futures):
                subset.extend(future.result())
        subset.sort(key=lambda x: x['Key'])
        if len(subset) == 0:
            continue
        for i in range(len(subset)):
            subset[i]['Category'] = dataset_name
        chunks.extend(subset)
    return chunks


def cost_aware_lrfu(ssd=True) -> Dict[str, List[Dict]]:
    cursor = mongo_operator.get_inactive_chunks(ssd)
    preemptive_chunks = defaultdict(list)
    max_crf = float('-inf')
    min_crf = float('inf')
    for preemptive_chunk in cursor:
        ref_times = preemptive_chunk["References"]
        cost = preemptive_chunk["Cost"]
        t_base = time.time()
        crf = 0.0
        for ref in ref_times:
            dur = t_base - ref.timestamp()
            af = float(evict_configs['attenuation']) + 1e-9
            crf += cost / dur * math.pow(1 / af, float(evict_configs['step']) * dur)
        loc = preemptive_chunk["Location"] if ssd else preemptive_chunk['SourceLocation']
        max_crf = max(max_crf, crf)
        min_crf = min(min_crf, crf)
        preemptive_chunk['CRF'] = crf
        preemptive_chunks[loc].append(preemptive_chunk)
    cursor.close()

    # Normalize the values in the dictionary
    for node, chunks in preemptive_chunks.items():
        for i in range(len(chunks)):
            preemptive_chunks[node][i]['CRF'] = min_crf / (1 + min_crf - max_crf)

    return preemptive_chunks


def download_chunks(worker_url: str, chunks: List[Dict], s3_operator: S3Operator, bucket_name: str, context: zmq.Context) -> Union[Dict, None]:
    for chunk in chunks:
        if 'ETag' in chunk:  # ETag is provided by S3 only
            etag = chunk['ETag'].strip('"')
        else:
            # read the first 1MB to generate hashtag is it's not provided by cloud service provider
            # this is for avoiding overflowing memory
            value = s3_operator.read_object(bucket_name, chunk['Key'], length=1048576)
            etag = utils.hashing(value)

        # some other jobs are preparing this item
        if etag in preparing_chunks:
            return None
        else:
            preparing_chunks.add(etag)

        now = datetime.utcnow().timestamp()
        try:
            last_modified = bson.timestamp.Timestamp(int(chunk['LastModified'].timestamp()), inc=1)
        except:
            last_modified = bson.timestamp.Timestamp(int(chunk['LastModified'].as_datetime().timestamp()), inc=1)
        
        chunk.update({
            "ETag": etag,
            "Bucket": bucket_name,
            "InitTime": bson.timestamp.Timestamp(int(now), inc=1),
            "LastModified": last_modified
        })

        # notify manager-worker to download file
        decompressed_file_size, download_latency = chunk['Size'], None
        logger.info(f"downloading item {etag} on node {chunk['SourceLocation']}...")

    req = pb.DownloadFileRequest()
    req.s3auth.CopyFrom(pb.S3Auth(**s3_operator.s3auth))
    req.bucket = bucket_name
    for chunk in chunks:
        req.keys.append(chunk['Key'])
        etag = chunk['ETag'].strip('"')
        req.destinations.append(f"{hdd_dir}/local/{etag}")

    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{worker_url}")
    socket.send_multipart([b"download", req.SerializeToString()])
    data = socket.recv()
    resp = pb.DownloadFileResponse.FromString(data)
    
    for i in range(len(chunks)):
        decompressed_file_size, download_latency = resp.sizes[i], resp.costs[i]
        chunks[i].update({
            "ChunkSize": decompressed_file_size,  # Size is ssd file size, ChunkSize is decompressed file size
            "DownloadLatency": int(1000 * download_latency),  # convert to millisecond
            "ExistOnHDD": True
        })
    return chunks


def chunk_extractor(node, worker_url: str, chunks: Dict, context=zmq.Context) -> int:
    req = pb.ExtractFileRequest()
    for chunk in chunks:
        key = chunk['Key']
        file_type = key.split('.')[-1].lower()
        if file_type in ['tar', 'bz2', 'zip', 'gz']:
            src = f"{hdd_dir}/local/{chunk['ETag']}"
            if chunk['Location'] == chunk['SourceLocation']:
                loc = 'local'
            else:
                loc = chunk['Location']
            dst = f"{ssd_dir}/{loc}/{chunk['ETag']}"
            
            chunk_path = f"{ssd_dir}/{chunk['Location']}/{chunk['ETag']}"
            if os.path.exists(chunk_path):
                # if 'ExtractionLatency' in chunk:
                #     return chunk['ExtractionLatency']
                # else:
                #     shutil.rmtree(dst)
                
                #--- this is for exp only
                shutil.rmtree(chunk_path)
            req.sources.append(src)
            req.destinations.append(dst)
    
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{worker_url}")
    socket.send_multipart([b'extract', req.SerializeToString()])
    data = socket.recv()
    resp = pb.ExtractFileResponse.FromString(data)
    logger.info(f"extracted {req.sources} to {req.destinations}")
    return node, [int(1000 * extract_latency) for extract_latency in resp.costs]


def redirect_chunk(chunk: Dict, sorted_nodes_list: List[str]) -> Dict:
    # Move the chunk to the first node that can accommodate it
    while True:
        for node in sorted_nodes_list:
            try:
                if node != chunk['Location']:
                    src_path = f"{ssd_dir}/{chunk['Location']}/"
                    dst_path = f"{ssd_dir}/{node}/"
                    if chunk['ChunkSize'] > utils.get_free_space(dst_path):
                        continue
                    resp = pb.MoveChunkRequest(src=src_path, dst=dst_path)
                    if resp.success:
                        mongo_operator.move_chunk(chunk['ETag'], node)
                        chunk['Location'] = node
                return chunk
            except OSError:  # handle the case that the node is out of space
                continue


def extract_chunks(workers_url: Dict[str, str], chunks: List[Dict]):
    futures = []
    context = zmq.Context()
    chunk_groups = defaultdict(list)
    for chunk in chunks:
        chunk_groups[chunk['SourceLocation']].append(chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for node in chunk_groups:
            futures.append(executor.submit(chunk_extractor, node, workers_url[node], chunk_groups[node], context))

    for future in concurrent.futures.as_completed(futures):
        node, costs = future.result()
        for i, cost in enumerate(costs):
            chunk_groups[node][i]['ExtractionLatency'] = cost
            chunk_groups[node][i]['ExistOnSSD'] = True
            
    processed_chunks = []
    for node in chunk_groups:
        processed_chunks.extend(chunk_groups[node])
    chunks = processed_chunks


def save_chunks_info(jobs: List[str], chunks: List[Dict]):
    for chunk in chunks:
        if 'References' not in chunk:
            chunk['References'] = []
        if 'Jobs' not in chunk:
            chunk['Jobs'] = jobs
        else:
            chunk['Jobs'].extend(jobs)
        
        if 'ExtractionLatency' in chunk and 'DownloadLatency' in chunk:
            chunk.setdefault('Cost', chunk['DownloadLatency'] + chunk['ExtractionLatency'])
        if chunk['ExistOnSSD']:
            if 'Status' in chunk:
                if chunk['Status']['code'] == ChunkStatus.ACTIVE:
                    chunk['Status']['active_count'] += 1
                else:
                    chunk['Status']['code'] = ChunkStatus.PENDING
            else:
                chunk['Status'] = {"code": ChunkStatus.ACTIVE, "active_count": 1}
        else:
            chunk['Status'] = {"code": ChunkStatus.PENDING, "active_count": 1}

        mongo_operator.replace_chunk(chunk['ETag'], chunk)
        if chunk['ETag'] in preparing_chunks:
            preparing_chunks.remove(chunk['ETag'])


def delete_preemptive_chunks(preemptive_chunks: List[Dict], require_space: int, ssd: bool = True) -> Tuple[bool, float]:
    # Each preemptive chunk has a CRF value and a ChunkSize, where the first can be considered as the cost of
    # deleting the chunk. Therefore, we can define the unit cost of the chunk as CRF/ChunkSize. In this case,
    # our target is to minimize the total cost. Here, we use a greedy method to delete chunks.
    for i in range(len(preemptive_chunks)):
        preemptive_chunks[i]['Value'] = preemptive_chunks[i]['CRF'] / preemptive_chunks[i]['ChunkSize']
    preemptive_chunks.sort(key=lambda chunk: chunk['Value'])

    released_space = 0
    while len(preemptive_chunks) > 0:
        delete_chunk = preemptive_chunks.pop(0)
        etag, size = delete_chunk['ETag'], delete_chunk['ChunkSize']
        if ssd:
            size = delete_chunk['ChunkSize']
            path = f"{ssd_dir}/{delete_chunk['Location']}/{etag}"
        else:
            size = delete_chunk['Size']
            path = f"{hdd_dir}/{delete_chunk['SourceLocation']}/{etag}"

        utils.delete_files(path)
        mongo_operator.delete_chunk(etag, delete_source=not ssd)
        released_space += size
        if released_space >= require_space:
            return True, released_space
        else:
            require_space -= size
    return False, released_space
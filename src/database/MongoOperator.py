import os
from pymongo.mongo_client import MongoClient
import json
import multiprocessing as mp
from collections import defaultdict
from typing import List, Dict, Any
from database import ChunkStatus
from configurations import ConfigParser
from logger import get_logger


logger = get_logger(level='DEBUG')


class MongoOperator:
    mongo_uri = None

    def __init__(self):
        parser = ConfigParser(components=['mongodb'])
        mongo_conn_conf = parser.get('mongodb')['client']
        mongo_uri = "mongodb://{}:{}@{}:{}".format(mongo_conn_conf['username'], mongo_conn_conf['password'],
                                                   mongo_conn_conf['host'], mongo_conn_conf['port'])
        mongo_client = MongoClient(mongo_uri)

        def load_collection(name, schema):
            collections = mongo_client[f"{mongo_conn_conf['db']}"].list_collection_names()
            with open(schema, 'r') as f:
                schema = json.load(f)
            if name not in collections:
                return mongo_client[f"{mongo_conn_conf['db']}"].create_collection(name=name,
                                                                                  validator={"$jsonSchema": schema},
                                                                                  validationAction="warn")
            else:
                return mongo_client[f"{mongo_conn_conf['db']}"][name]

        if os.path.exists('database/schemas'):
            schema_dir = 'database/schemas'
        elif os.path.exists('../database/schemas'):
            schema_dir = '../database/schemas'
        else:
            raise FileNotFoundError("The mongodb schema folder does not exist.")
        
        self.client_col = load_collection('Client', f"{schema_dir}/client.json")
        self.job_col = load_collection('Job', f"{schema_dir}/job.json")
        self.dataset_col = load_collection('Datasets', f"{schema_dir}/datasets.json")
        

    """-------------Operation for the collection: Client-------------"""

    def find_user(self, username: str, password=None):
        if password is None:
            return self.client_col.find_one(filter={"Username": username})
        else:
            return self.client_col.find_one(filter={"$and": [{"Username": username, "Password": password}]})

    def get_s3auth(self, **kwargs):
        return self.find_user(kwargs['username'], kwargs['password'])['S3Auth']

    def update_user_auth(self, username: str, auth: str):
        return self.client_col.update_one(filter={"Username": username}, update={"$set": {"S3Auth": auth}})

    def add_user(self, username: str, password: str, s3auth: Dict):
        result = self.client_col.insert_one(
            {"Username": username, "Password": password, "S3Auth": s3auth, "Status": True})
        return result.acknowledged

    def disconnect_user(self, username, password):
        result = self.client_col.update_one(
            {
                "Username": username,
                "Password": password,
            },
            {"$set": {"Status": True, "Jobs": []}}
        )
        return result.matched_count > 0

    """-------------Operation for the collection: Job-------------"""

    def add_job(self, username: str, job_id: str, dltdeploy_id: str, num_workers: int, datasource: Dict, resource_requests: Dict, etags: Dict):
        job_info = {
            "Meta": {
                "Username": username,
                "JobId": job_id,
                "DLTDeployId": dltdeploy_id,
                "NumWorkers": num_workers,
                "Datasource": datasource,
                "ResourceRequests": resource_requests
            },
            "Status": "pending",
            "ChunkETags": etags
        }
        self.job_col.insert_one(job_info)

    def delete_job(self, job_id):
        pipeline = [
            {"$addFields": {"MetaId": {"$concat": ["$Meta.DLTDeployId", "-", "$Meta.JobId"]}}},
            {"$match": {"MetaId": job_id}}
        ]
        result = self.job_col.delete_one({'$and': pipeline})
        if result.deleted_count == 0:
            return False
        else:
            return True

    def find_job(self, job_id):
        pipeline = [
            {"$sort": {"_id": -1}},
            {"$addFields": {"MetaId": {"$concat": ["$Meta.DLTDeployId", "-", "$Meta.JobId"]}}},
            {"$match": {"MetaId": job_id}}
        ]
        result_cursor = self.job_col.aggregate(pipeline)
        doc = next(result_cursor, None)
        if doc is None:
            return None
        else:
            return doc

    def update_job(self, job_id, update, operation='$set'):
        pipeline = [
            {"$addFields": {"MetaId": {"$concat": ["$Meta.DLTDeployId", "-", "$Meta.JobId"]}}},
            {"$match": {"MetaId": job_id}}
        ]
        job_document = self.job_col.aggregate(pipeline).next()
        if job_document:
            result = self.job_col.update_one({'_id': job_document['_id']}, {operation: update})
            return result.matched_count > 0
        else:
            raise RuntimeError(f"failed to update job {job_id}, operation: {str(update)}")
            return False
    
    """-------------Operation for the collection: Dataset-------------"""

    def filter_etags(self, etags: List):
        return self.dataset_col.aggregate(pipeline=[{"$match": {"ETag": {"$in": etags}}}])

    def project_etags(self, etags: List):
        return self.dataset_col.aggregate([
            {"$match": {"ETag": {"$in": etags}}},
            {"$project": {"Key": 1, "Location": 1, "ETag": 1, "Blocks": 1, "Size": 1, "_id": 0}}
        ])

    def find_etag(self, etag: str):
        return self.dataset_col.find_one({"ETag": etag})

    def get_preemptive_chunks(self):
        chunks = defaultdict(list)
        results = self.dataset_col.find({"Status": ChunkStatus.INACTIVE})
        for doc in results:
            chunks[doc['Location']].append(doc)

        for node in list(chunks.keys()):
            chunks[node] = sorted(chunks[node], key=lambda item: item['ChunkSize'])
        return chunks

    def get_inactive_chunks(self, ssd=False):
        if ssd:
            return self.dataset_col.find({'Status': ChunkStatus.INACTIVE, 'ExistOnSSD': True})
        else:
            return self.dataset_col.find({'Status': ChunkStatus.INACTIVE, 'ExistOnHDD': True})

    def add_chunk(self, chunk):
        self.dataset_col.insert_one(chunk)

    def add_chunks(self, chunks):
        self.dataset_col.insert_many(chunks)

    def delete_chunks(self, etags: List):
        self.dataset_col.delete_many({"ETag": {"$in": etags}})

    def delete_chunk(self, etag, delete_source=False):
        if delete_source:
            self.dataset_col.delete_one({"ETag": etag}, {"$set": {"ExistOnSSD": False, "ExistOnHDD": False}})
        else:
            self.dataset_col.update_one({"ETag": etag}, {"$set": {"ExistOnSSD": False}})

    def move_chunk(self, etag: str, new_loc: str):
        self.dataset_col.update_one({"ETag": etag}, {"$set": {"Location": new_loc}})

    def replace_chunk(self, etag, new_chunk):
        self.dataset_col.replace_one({"ETag": etag}, new_chunk, upsert=True)
    
    def update_chunk(self, etag, update):
        self.dataset_col.update_one({"ETag": etag}, update)

class AsyncMongoOperator(MongoOperator):
    def __init__(self):
        super().__init__()
        self.started = False
        self.queue = mp.Queue()
        self.queue.cancel_join_thread()

    def start(self):
        self.proc = mp.Process(target=self.__apply, daemon=True)
        self.proc.start()
        self.started = True

    def execute(self, col: int, func: str, args: Any):
        self.queue.put((col, func, args))

    def __apply(self):
        while True:
            try:
                col, function, args = self.queue.get()
                col = [self.client_col, self.job_col, self.dataset_col][col]
                func = {'update_many': col.update_many}[function]
                func(*args)
            except Exception as ex:
                logger.error(msg=str(ex))

    def terminate(self):
        self.proc.terminate()
        self.started = False
        del self.proc

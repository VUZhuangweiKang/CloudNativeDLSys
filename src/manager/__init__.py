from logger import get_logger
from configurations import ConfigParser
from databus import dbus_pb2 as pb
from cloudbucket.CloudBucketOperator import S3Operator
from commons import utils
import numpy as np
import os
from typing import List, Dict, Union, Tuple
from database import MongoOperator
from k8s.K8sOperator import DLTDeployOperator


logger = get_logger(name=__name__, level='debug')
parser = ConfigParser(components=['commons', 'manager'])
data_config = parser.get('commons')['data']
manager_config = parser.get('manager')
hdd_dir = data_config['hdd_base_dir']
ssd_dir = data_config['ssd_base_dir']
evict_configs = manager_config['data_eviction']
schedule_freq = int(manager_config['scheduler']['frequency'])
MAX_BANDWIDTH = 125000000  # Bps, = 1Gbps

mongo_operator = MongoOperator.MongoOperator()
# avoid preparing duplicated chunks for different jobs, set is thread-safe
preparing_chunks = set()
k8s_operator = DLTDeployOperator()
os.environ["H2O_AUTOENCODER_VERBOSE"] = "0"


class JobQueueItem:
    def __init__(self, name: str, job_id: str, dltdeploy_id: str, cred: Dict[str, Union[str, Dict]], spec: Dict, chunks: List[Dict],
                 storage_gap: Tuple[float] = (np.inf, np.inf), peers: List[str] = None):
        self.name = name.lower()
        self.dltdeploy_id = dltdeploy_id.lower()
        self.job_id = job_id.lower()
        self.cred = cred
        self.spec = spec
        self.chunks = chunks

        if peers:
            self.peers = peers
        else:
            self.peers = [job_id]
        self.storage_gap = storage_gap
        self.deployable = None
        self.nodes = []
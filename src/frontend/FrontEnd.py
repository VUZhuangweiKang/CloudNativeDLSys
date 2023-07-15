import json
import jsonschema
from flask import Flask, request
import zmq
from typing import Dict, List

from cloudbucket.CloudBucketOperator import S3Operator
from database.MongoOperator import MongoOperator
from databus import dbus_pb2 as pb
from commons import utils
from logger import get_logger
from configurations import ConfigParser


app = Flask(__name__)
logger = get_logger(__name__)
parser = ConfigParser(components=['dbus'])
dbus_configs = parser.get('dbus')['manager']
manager_uri = f"{dbus_configs['hostname']}:{dbus_configs['port']}"
context = zmq.Context()
mongo_operator = MongoOperator()


def prepare_credential(cred):
    # Prepare S3Auth
    s3auth = cred['s3auth']
    pb_s3auth = pb.S3Auth()
    pb_s3auth.aws_access_key_id = s3auth['aws_access_key_id']
    pb_s3auth.aws_secret_access_key = s3auth['aws_secret_access_key']
    pb_s3auth.region_name = s3auth['region_name']

    # Prepare Credential
    pb_cred = pb.Credential()
    pb_cred.username = cred["username"]
    pb_cred.password = cred["password"]
    pb_cred.s3auth.CopyFrom(pb_s3auth)
    return pb_cred


def auth_client(cred: pb.Credential, conn_check: bool = False) -> pb.RC:
    result = mongo_operator.find_user(cred.username)
    if result is None:
        rc = pb.RC.NO_USER
    else:
        if cred.password == result['Password']:
            if conn_check:
                rc = pb.RC.CONNECTED if result['Status'] else pb.RC.DISCONNECTED
            else:
                rc = pb.RC.CONNECTED
        else:
            rc = pb.RC.WRONG_PASSWORD

    # check whether to update s3 auth information
    s3auth = utils.protobuf_msg_to_dict(cred.s3auth)
    if rc == pb.RC.CONNECTED and s3auth is not None and result['S3Auth'] != s3auth:
        result = mongo_operator.update_user_auth(cred.username, s3auth)
        if result.modified_count != 1:
            logger.error("user {} is connected, but failed to update S3 authorization information.".format(cred.username))
            rc = pb.RC.FAILED
    return rc


@app.route('/connect', methods=['POST'])
def connect():
    data = request.json
    try:
        with open('schemas/credential.json') as f:
            schema = json.load(f)
        jsonschema.validate(instance=data, schema=schema)
        
        # Prepare ConnectRequest
        cred = prepare_credential(data)
        createUser = True
        auth_rc = auth_client(cred=cred)
        
        if auth_rc == pb.RC.WRONG_PASSWORD:
            rc = pb.RC.FAILED
            resp = "wrong password"
        elif auth_rc == pb.RC.NO_USER:
            if createUser:
                try:
                    if mongo_operator.add_user(cred.username, cred.password, utils.protobuf_msg_to_dict(cred.s3auth)):
                        logger.info(f"user {cred.username} connected")
                        rc = pb.RC.CONNECTED
                        resp = "connection setup"
                    else:
                        raise Exception
                except Exception:
                    rc = pb.RC.FAILED
                    resp = "connection error"
            else:
                rc = pb.RC.FAILED
                resp = f"not found user {cred.username}"
        elif auth_rc == pb.RC.DISCONNECTED:
            if mongo_operator.disconnect_user(cred.username, cred.password):
                rc = pb.RC.CONNECTED
                resp = "connection recreated"
                logger.info(f"user {cred.username} connected")
            else:
                rc = pb.RC.FAILED
                resp = "connection error"
        else:
            rc = pb.RC.CONNECTED
            resp = "connection setup"
            logger.info(f"user {cred.username} connected")
        
        return resp, 200 if rc == pb.RC.CONNECTED else 404
    except jsonschema.exceptions.ValidationError as e:
        return e.message, 400


def validate_datasource(s3_operator: S3Operator, bucket_name: str, datasource: Dict[str, Dict[str, List]]):
    keys = []
    for _, k in datasource.items():
        keys.extend(k)
    invalid_keys = [1 - s3_operator.isvalid_key(bucket_name, k) for k in keys]
    return sum(invalid_keys) == 0


@app.route('/deploy', methods=['POST'])
def deploy():
    data = request.json
    try:
        with open('schemas/dltdeploy.json') as f:
            schema = json.load(f)
        jsonschema.validate(instance=data, schema=schema)
        cred = data['credential']
        s3_operator = S3Operator(mongo_operator.get_s3auth(**cred))

        # User specifies the datasource under the dltdeployment specification. In this case, we assume all jobs are using
        # the same datasource. If user also specifies the datasource field under a job specification,
        # the inner datasource specification will override the outside datasource. When gang-scheduling is enabled,
        # we assume all jobs have the same datasource. The use cases for Gang-scheduling include hyperparameter
        # tuning and multitask learning, where users want to save the tuning time by deploy multiple jobs together.
        global_ds = None
        if 'datasource' in data:
            global_ds = data['datasource']
            if not validate_datasource(s3_operator, global_ds['bucket'], global_ds['keys']):
                return "Failed to validate data source", 404
        elif data['gangScheduling']:
            return "Data source must be specified, when gang-scheduling is enabled.", 404

        for i in range(len(data['jobs'])):
            job = data['jobs'][i]
            if 'datasource' not in job:
                data['jobs'][i]['datasource'] = global_ds
            else:
                # individual datasource under each job
                ds = job['datasource']
                if not validate_datasource(s3_operator, ds['bucket'], ds['keys']):
                    return "Failed to validate data source", 404
        
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{manager_uri}")
        req = pb.DeployRequest()
        req.name = data['name'].lower()
        req.credential.CopyFrom(prepare_credential(cred))
        req.gangScheduling = data['gangScheduling']
        req.jobs = json.dumps(data['jobs'])
        socket.send_multipart([b"deploy", req.SerializeToString()])
        data = socket.recv()
        resp = pb.DeployResponse.FromString(data)
        if resp.rc == pb.RC.DEPLOYED:
            return "success", 200
        else:
            return resp.resp, 500
    except jsonschema.exceptions.ValidationError as e:
        return e.message, 400

@app.route('/dltdeploys/<string:username>', methods=['GET'])
def list_dltdeployments(username):
    pass

@app.route('/dltjobs/<string:username>/<string:dltdeploy>', methods=['GET'])
def list_jobs(username, dltdeploy):
    pass


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

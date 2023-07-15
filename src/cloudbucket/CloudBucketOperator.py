import boto3
import botocore
import os


def get_s3_client(s3auth):
    session = boto3.Session(**s3auth)
    s3_client = session.client('s3')
    return s3_client


class S3Operator:
    def __init__(self, s3auth):
        self.s3auth = s3auth
        self.client = get_s3_client(s3auth)

    def paginate(self, bucket, prefix):
        paginator = self.client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        results = []
        for page in pages:
            if 'Contents' not in page:
                continue
            else:
                results.append(page['Contents'])
        return results

    def read_object(self, bucket, key, length=None):
        if length is None:
            return self.client.get_object(Bucket=bucket, Key=key)['Body'].read()
        else:
            return self.client.get_object(Bucket=bucket, Key=key, Range='bytes=0-{}'.format(length))['Body'].read()

    def isvalid_key(self, bucket, key):
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=key)
        return 'Contents' in response

    def download_obj(self, bucket, key, destination):
        try:
            if not os.path.exists(destination):
                self.client.download_file(bucket, key, destination)
            return True
        except botocore.exceptions.ClientError as e:
            return False

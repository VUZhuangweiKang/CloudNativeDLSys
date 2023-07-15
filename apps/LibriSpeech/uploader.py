# This file should be placed in the ~/LibriSpeech/LibriSpeech_dataset foler

import boto3
import glob
import multiprocessing
import concurrent.futures

session = boto3.Session()
s3 = session.client("s3")
bucket = 'vuzhuangwei'


def upload_objects():
    upload = lambda path: s3.upload_file(path, bucket, 'LibriSpeech/{}'.format(path))
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for folder in ['train', 'val', 'test']:
            wav_files = glob.glob('{}/wav/*'.format(folder))
            txt_files = glob.glob('{}/txt/*'.format(folder))
            wav_files.sort()
            txt_files.sort()
            for i in range(len(wav_files)):
                futures.append(executor.submit(upload, wav_files[i]))
                futures.append(executor.submit(upload, txt_files[i]))
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    upload_objects()
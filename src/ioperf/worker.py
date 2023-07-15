import time
import zmq
import os
import logging
import numpy as np
import shutil
from multiprocessing import cpu_count, Process


def get_logger(name=__name__, level: str = 'INFO', file=None):
    levels = {"info": logging.INFO, "error": logging.ERROR, "debug": logging.DEBUG}
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(levels[level.lower()])

    cl = logging.StreamHandler()
    cl.setLevel(levels[level.lower()])
    cl.setFormatter(formatter)
    logger.addHandler(cl)

    if file is not None:
        fl = logging.FileHandler(file)
        fl.setLevel(levels[level.lower()])
        fl.setFormatter(formatter)
        logger.addHandler(fl)
    return logger


def generate_files(dir_path, total_size, file_size):
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 计算需要生成的文件数量
    num_files = total_size // file_size

    # 定义一个函数，每个进程将运行此函数来生成文件
    def create_files(start, end, dir_path, file_size):
        for i in range(start, end):
            # 创建一个包含随机数据的字符串
            data = os.urandom(file_size)

            # 创建文件路径
            file_path = os.path.join(dir_path, f"file_{i}.bin")

            # 写入文件
            with open(file_path, "wb") as f:
                f.write(data)

    # 计算每个进程应该生成多少个文件
    num_processes = cpu_count()
    files_per_process = num_files // num_processes

    processes = []

    # 创建并启动进程
    for i in range(num_processes):
        start = i * files_per_process
        # 最后一个进程处理剩下的所有文件
        end = start + files_per_process if i < num_processes - 1 else num_files
        process = Process(target=create_files, args=(start, end, dir_path, file_size))
        process.start()
        processes.append(process)

    # 等待所有进程完成
    for process in processes:
        process.join()


def read_files(dir_path):
    # 遍历文件夹中的所有文件
    all_files = os.listdir(dir_path)
    np.random.shuffle(all_files)
    start = time.time()
    for filename in all_files:
        file_path = os.path.join(dir_path, filename)

        # 只处理文件，忽略文件夹
        if os.path.isfile(file_path):
            # 打开并读取文件
            with open(file_path, "rb") as f:
                data = f.read()
    end = time.time()
    return end-start


if __name__ == '__main__':
    logger = get_logger()
    controller = os.getenv('CONTROLLER')
    assert controller is not None
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:5555")

    while True:
        rcv_str = socket.recv_string()
        io_mode, total_size, block_size = rcv_str.split(' ')
        
        if io_mode == '0':
            data_dir = "/mnt/local/data"
        else:
            data_dir = "/mnt/nfs/data"
        
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)

        generate_files(data_dir, int(total_size), int(block_size))
        
        os.system(f"vmtouch -e {data_dir}")
        
        latency = read_files(data_dir)
        
        shutil.rmtree(data_dir)
        socket.send_string(f"{latency}")

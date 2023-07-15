import os
import sys
from multiprocessing import Process, cpu_count


total_size, file_size, data_dir = sys.argv[1:]

# 存放文件的目录
dir_path = f"{data_dir}/{total_size}_{file_size}"

def parse_input():
    global total_size, file_size
    # 每个文件的大小
    if file_size.endswith('GB'):
        file_size = int(file_size.replace('GB', '')) * 1024 * 1024 * 1024
    elif file_size.endswith('MB'):
        file_size = int(file_size.replace('MB', '')) * 1024 * 1024
    elif file_size.endswith('KB'):
        file_size = int(file_size.replace('KB', '')) * 1024
    elif file_size.endswith('B'):
        file_size = int(file_size.replace('B', ''))
    else:
        file_size = int(file_size)

    # 要生成的总数据量，单位是字节
    if total_size.endswith('GB'):
        total_size = int(total_size.replace('GB', '')) * 1024 * 1024 * 1024
    elif total_size.endswith('MB'):
        total_size = int(total_size.replace('MB', '')) * 1024 * 1024
    elif total_size.endswith('KB'):
        total_size = int(total_size.replace('KB', '')) * 1024
    elif total_size.endswith('B'):
        total_size = int(total_size.replace('B', ''))
    else:
        total_size = int(total_size)
    
    return total_size, file_size

total_size, file_size = parse_input()

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

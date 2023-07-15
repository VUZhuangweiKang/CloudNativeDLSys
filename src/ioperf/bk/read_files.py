import os
import sys
import time
import numpy as np


total_size, file_size, data_dir = sys.argv[1:]

# 要读取的文件夹路径
dir_path = f"{data_dir}/{total_size}_{file_size}"

# 遍历文件夹中的所有文件
all_files = os.listdir(dir_path)
np.random.shuffle(all_files)
for filename in all_files:
    file_path = os.path.join(dir_path, filename)

    # 只处理文件，忽略文件夹
    if os.path.isfile(file_path):
        # 打开并读取文件
        with open(file_path, "rb") as f:
            data = f.read()
#!/bin/bash

# 要保存结果的文件路径
output_file="results.txt"

# 清空结果文件
echo "" > $output_file

# 文件生成和读取的目录
dir_path="data"

total_sizes=(32 64 128)  # GB
file_sizes=(128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)  # KB

# 总文件大小范围
for total_size in "${total_sizes[@]}"; do
    # 单独文件大小范围
    for file_size in "${file_sizes[@]}"; do

        # 使用Python脚本生成文件
        python3 generate_files.py ${total_size}GB ${file_size}KB $dir_path

        vmtouch -e /mnt/local/data/
        vmtouch -e /mnt/nfs/data/
        
        # 获取开始时间
        start_time=$(date +%s.%N)
        
        # 使用Python脚本读取文件
        python3 read_files.py ${total_size}GB ${file_size}KB $dir_path

        # 获取结束时间并计算总时间
        end_time=$(date +%s.%N)
        total_time=$(echo "$end_time - $start_time" | bc)

        # 将结果追加到结果文件中
        echo "Total size: ${total_size}GB, File size: ${file_size}KB, Total time: ${total_time} seconds" >> $output_file

        # 删除生成的文件
        rm -r $dir_path
    done
done
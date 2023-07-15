import os


def get_cpu_free_mem():
    total, used, free, shared, cache, available = map(int, os.popen('free -t -m').readlines()[1].split()[1:])
    return free


# import nvidia_smi
# def get_gpu_free_mem():
#     try:
#         nvidia_smi.nvmlInit()
#         deviceCount = nvidia_smi.nvmlDeviceGetCount()
#         total = 0
#         total_free = 0
#         total_used = 0
#         for i in range(deviceCount):
#             handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
#             info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#             total += info.total
#             total_free += info.total_free
#             total_used += info.total_used
#         return total_free
#     except:
#         return 0


def count_files(dir):
    count = 0
    for root_dir, cur_dir, files in os.walk(dir):
        count += len(files)
    return count

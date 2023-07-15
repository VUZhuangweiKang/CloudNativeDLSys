import numpy as np
import glob
import os, sys


if __name__=="__main__":
    r, c, b = tuple(sys.argv[1:])
    r, c, b = int(r), float(c), int(b)
    dir = "data/run{}/{}/{}".format(r, c, b)
    load_time = np.sum(np.load('{}/load_time.npy'.format(dir)))
    if os.path.exists('{}/train_cache_usage.npy'.format(dir)):
        avg_cache_usage = np.load('{}/train_cache_usage.npy'.format(dir))
        avg_cache_usage = np.mean(avg_cache_usage, dtype=int) / b
        cache_hits = 0
        for ch in glob.glob("{}/cache_hits*".format(dir)):
            cache_hits += np.load(ch)
        cache_hit_rate = 100 * cache_hits / (b * 100)
    print('Summary: \nload_time(s): {}\tcache_usage(batch): {}\tcache_hit_rate(%): {}'.format(load_time, avg_cache_usage, cache_hit_rate))
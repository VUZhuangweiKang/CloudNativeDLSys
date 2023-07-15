import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DeepSpeech Training')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--mini-batches', default=1000, type=int, metavar="N", 
                        help="the number of mini-batches for each epoch")
    parser.add_argument('--sim-compute-time', type=float, default=0.5,
                    help='simulated computation time per batch in second')

    args = parser.parse_args()
    
    with open('/app/dlcache_exp.txt', "w") as f:
        f.write(f"{args.workers},{args.batch_size},{args.epochs},{args.mini_batches},{args.print_freq},{args.sim_compute_time},0")
    
    os.system("python3 train.py +configs=librispeech")
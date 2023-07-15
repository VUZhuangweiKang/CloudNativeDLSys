import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from lib.DLCJob import *
from CCDataset import *
import argparse


# 3. 自定义 collate_fn 函数，将不同长度的文本序列转化为相同长度
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels)
    return texts, labels

# 4. 定义模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.rnn = nn.RNN(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.rnn(embedded)
        return self.fc(output[:, -1, :])  # 只使用最后一个时间步的输出


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Common Crawler Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--mini-batches', default=0, type=int, metavar="N", 
                        help="the number of mini-batches for each epoch")
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--sim-compute-time', type=float, default=0.5,
                        help='simulated computation time per batch in second')

    args = parser.parse_args()
    
    dataset = TextDataset(name="train")
    data_loader = DLCJobDataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    # 5. 训练模型
    model = RNNClassifier(len(vocab), embed_dim=64, hidden_dim=128, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        for texts, labels in data_loader:
            time.sleep(args.sim_compute_time)
            # optimizer.zero_grad()
            # outputs = model(texts)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

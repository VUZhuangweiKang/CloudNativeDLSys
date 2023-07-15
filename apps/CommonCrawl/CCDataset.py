from typing import Any
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from lib.DLCJob import DLCJobDataset
import io
import json


# 1. Tokenizer 和词汇表
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open('text.txt', encoding="utf8"))))


# 2. 定义处理函数
def preprocess(text):
    tokens = tokenizer(text)
    # 转化为整数序列
    token_ids = [vocab[token] for token in tokens]
    return torch.tensor(token_ids)


class TextDataset(DLCJobDataset):
    def __init__(self, name: str = 'train'):
        super().__init__(name)
        self.cls_idx = {}
        
    def _process_item(self, item_cloud_path: str, contents: Any) -> Any:
        sample = json.loads(contents)
        text, label = sample['text'], sample['label']
        text = preprocess(text)
        return text, label
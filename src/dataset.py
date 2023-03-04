from src import data_processors
import torch
from argparse import Namespace


class DumbTokenizer(object):
    def __init__(self, chars: list):
        self.chars = chars
        self.vocab_size = len(self.chars)

        stoi = {ch: i for i, ch in enumerate(self.chars)}
        itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [stoi[c] for c in s]  # Encoder takes string and outputs a list of ints
        self.decode = lambda l: "".join([itos[i] for i in l])  # Decoder takes a list of integers and outputs a string


class DumbDataset(object):
    def __init__(self, split: str, args: Namespace, split_ratio: float = 0.9):
        self.split = split
        self.block_size = args.block_size
        self.batch_size = args.batch_size
        self.device = args.device

        text = data_processors.store_proced_data()

        chars = sorted(list(set(text)))
        tokenizer = DumbTokenizer(chars)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # Train/test split
        n = int(split_ratio * len(data))
        train_data = data[:n]
        val_data = data[n:]
        self.data = train_data if split == "train" else val_data
        self.tokenizer = tokenizer

    def get_batch(self):
        # Generate small batch of inputs x and targets y
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


def get_datasets(args: Namespace):
    return tuple(DumbDataset(s, args) for s in ("train", "val"))

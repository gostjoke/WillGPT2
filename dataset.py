from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import tiktoken
import config

class TextDataset(Dataset):

    def __init__(self):

        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )

        text = "\n".join(dataset["text"])

        enc = tiktoken.get_encoding("gpt2")

        tokens = enc.encode(text)

        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - config.block_size

    def __getitem__(self, idx):

        x = self.data[idx:idx+config.block_size]

        y = self.data[idx+1:idx+config.block_size+1]

        return x, y
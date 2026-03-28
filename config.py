# config.py

model_save_version = 2
vocab_size = 50257

block_size = 128

n_layer = 8
n_head = 8
n_embd = 512

batch_size = 32
learning_rate = 3e-4

max_iters = 5000

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
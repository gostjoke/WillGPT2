uv pip install torch --torch-backend=auto

# WillGPT2

A minimal GPT-style language model implemented in **PyTorch**.  
This project trains a small transformer language model on the **WikiText-2 dataset** and supports **CUDA GPU acceleration**.

The goal of this project is educational: to demonstrate how a GPT-style model can be implemented and trained from scratch with relatively little code.

---

# Features

- Minimal GPT architecture
- PyTorch implementation
- Training on WikiText-2 dataset
- CUDA GPU support
- Tokenization using `tiktoken`
- Simple text generation

---

# Model Architecture

The model is a simplified GPT-style transformer including:

- Token embeddings
- Positional embeddings
- Multi-head causal self-attention
- Feed-forward layers
- Layer normalization
- Language modeling head

Main components:
Embedding
↓
Transformer Blocks (N layers)
↓
LayerNorm
↓
Linear Head
↓
Token Prediction

# Dataset
WikiText-2
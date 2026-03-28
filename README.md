uv pip install torch --torch-backend=auto

# WillGPT2
Tien-Wei Hsu First trained LLM

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

### version 2
add checkpoint and improve the train process, add drop

QK^T
 ↓
softmax
 ↓
dropout      ← 這裡
 ↓
attention @ V
 ↓
linear projection
 ↓
dropout      ← 這裡

### 作用是什麼？

Dropout 在 Attention 裡主要是：

防止 overfitting
讓模型不要過度依賴某些 token
提高泛化能力

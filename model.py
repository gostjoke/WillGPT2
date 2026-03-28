import torch
import torch.nn as nn
import config
import math
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1,1,config.block_size,config.block_size)
        )

    def forward(self, x):

        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)

        att = att.masked_fill(
            self.mask[:,:,:T,:T] == 0,
            float('-inf')
        )

        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att)  # guard against all-masked rows producing NaN

        y = att @ v

        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.proj(y)

        return y

class Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = CausalSelfAttention()

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd)
        )

    def forward(self, x):

        x = x + self.attn(self.ln1(x))

        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.pos_emb = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.blocks = nn.Sequential(
            *[Block() for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )
    
    def forward(self, idx, targets=None):

        B,T = idx.shape

        tok_emb = self.token_emb(idx)

        pos = torch.arange(0,T,device=idx.device)
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.head(x)

        loss = None

        if targets is not None:

            logits = logits.view(-1, config.vocab_size)
            targets = targets.view(-1)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
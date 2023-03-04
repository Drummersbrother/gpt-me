from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, train_ds, val_ds, args: Namespace):
    out = {}
    model.eval()
    for split_name, split_ds in (("train", train_ds), ("val", val_ds)):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = split_ds.get_batch()
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size: int, args: Namespace):
        super().__init__()
        self.key = nn.Linear(args.n_embed, head_size, bias=False)
        self.query = nn.Linear(args.n_embed, head_size, bias=False)
        self.value = nn.Linear(args.n_embed, head_size, bias=False)
        # not trainable, assign as buffer
        self.register_buffer("tril", torch.tril(torch.ones(args.block_size, args.block_size)))

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B, T, C
        q = self.query(x)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Perform weighted aggregation of values
        v = self.value(x)  # B, T, C
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of attention in parallel"""

    def __init__(self, num_heads: int, head_size: int, args: Namespace):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, args) for _ in range(num_heads)])
        self.proj = nn.Linear(args.n_embed, args.n_embed)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd: int, args: Namespace):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd),
                                 nn.Dropout(args.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd: int, n_head: int, args: Namespace):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, args)
        self.ffwd = FeedForward(n_embd, args)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # per-token layernorm
        x = x + self.ffwd(self.ln2(x))  # per-token layernorm
        return x


class BigramLanguageModel(nn.Module):
    """
    Bigram model
    """

    def __init__(self, args: Namespace):
        super().__init__()
        # Each token reads off the token embeddings from the lookup table
        self.token_embedding_table = nn.Embedding(args.vocab_size, args.n_embed)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embed)
        self.blocks = nn.Sequential(*[Block(args.n_embed, args.n_head, args) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embed)
        self.lm_head = nn.Linear(args.n_embed, args.vocab_size)
        self.args = args

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both [B, T] tensors of integers
        tok_emb = self.token_embedding_table(idx)  # [B, T, C] batch, time, channel (n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.args.device))  # (T, C)
        x = tok_emb + pos_emb  # broadcasting works out, right-aligning and the new dimension comes to pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # Need to reshape to match shape that pytorch wants NLL/CE loss in
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is [B, T] tensor of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.args.block_size:]
            # get preds
            logits, loss = self(idx_cond)
            # Focus on last timestep
            logits = logits[:, -1, :]  # [B, C]
            # Get probabilites by softmax
            probs = F.softmax(logits, dim=-1)  # [B, C] still
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # [B, T+1]
        return idx

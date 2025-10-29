"""
A minimal Transformer (encoder-only) built with PyTorch Lightning.

Task: toy sequence classification — predict whether a special token appears in the input.
- No external datasets, runs on CPU in minutes.
- Clear, step-by-step comments for each module.

Requirements (install once):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install pytorch-lightning==2.4.0

Run:
    python pl_simple_transformer.py --max_epochs 5 --batch_size 128 --d_model 128

Tip: increase --max_epochs for better accuracy, or tweak vocab/sequence lengths.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# -------------------------------
# Utilities: reproducibility
# -------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# -------------------------------
# Positional Encoding (sinusoidal)
# -------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Classic sine/cosine positional encoding from 'Attention Is All You Need'.

    Adds a deterministic vector to each token embedding to inject order info.
    Shape: input [B, T, D] -> output [B, T, D]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        # Pre-compute pe[0:max_len] once, register as buffer (no grad, saved with state)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [T, D]

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # broadcast to [B, T, D]


# -------------------------------
# Attention building blocks
# -------------------------------
class ScaledDotProductAttention(nn.Module):
    """Single scaled dot-product attention head.

    - q, k, v: [B, H, T, Dh] (H=heads, Dh=d_model/H)
    - mask:    [B, 1, 1, T] with 0 for keep and -inf for pad (added to logits)
    Returns:   [B, H, T, Dh]
    """
    def __init__(self):
        super().__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None) -> Tensor:
        d_k = q.size(-1)
        # Raw attention scores: [B, H, Tq, Tk]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores + attn_mask  # add -inf to padding positions
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention (no bias for clarity).

    input:  x [B, T, D]
    output: [B, T, D]
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttention()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B, T, D] -> [B, H, T, Dh]
        B, T, D = x.size()
        x = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        # [B, H, T, Dh] -> [B, T, D]
        B, H, T, Dh = x.size()
        x = x.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return x

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        # Project to Q, K, V and split heads
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        # Build additive mask for pads: [B, 1, 1, T] with 0 or -inf
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(x.dtype)
            attn_mask = (1.0 - attn_mask)  # 1 for pad->0 keep; invert to 0 keep, 1 pad
            attn_mask = attn_mask.masked_fill(attn_mask > 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
            # Wait, above is confusing; simpler: key_padding_mask: 1 for pad, 0 for real
            # We'll rebuild properly below.
        if key_padding_mask is not None:
            # key_padding_mask: [B, T], True/1 where PAD
            # We want: [B, 1, 1, T] -> 0 for valid, -inf for PAD
            mask = key_padding_mask[:, None, None, :]
            attn_mask = torch.zeros_like(mask, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(mask, float("-inf"))

        # Attention per head
        context = self.attn(q, k, v, attn_mask)
        # Merge heads and final projection
        out = self._merge_heads(context)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network (two linear layers + GELU)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """One Transformer encoder block with pre-norm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        # Pre-norm + residual
        attn_out = self.self_attn(self.ln1(x), key_padding_mask)
        x = x + attn_out
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        return x


class TransformerEncoder(nn.Module):
    """Stack of encoder layers."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, d_ff: int = 256, max_len: int = 256, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [B, T] token ids
        key_padding_mask = (x == self.pad_id)  # [B, T] True where PAD
        h = self.embed(x)
        h = self.pos(h)
        for layer in self.layers:
            h = layer(h, key_padding_mask)
        h = self.ln(h)
        return h, key_padding_mask


class CLSClassifier(nn.Module):
    """Classification head using a learned [CLS] token at position 0."""
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, h: Tensor) -> Tensor:
        # h: [B, T, D]; we take the first token representation
        cls_h = h[:, 0, :]
        return self.fc(cls_h)


# -------------------------------
# Synthetic dataset for quick demos
# -------------------------------
@dataclass
class ToyConfig:
    vocab_size: int = 100
    seq_len: int = 32
    min_len: int = 8
    target_token: int = 2  # if present anywhere, label=1, else 0
    pad_id: int = 0
    cls_id: int = 1


class ToySeqDataset(Dataset):
    """Generates random token sequences.

    Rule: label=1 if target_token appears at least once; otherwise 0.
    The [CLS] token is *prepended* later in the collate function.
    """
    def __init__(self, n_samples: int, cfg: ToyConfig):
        self.n = n_samples
        self.cfg = cfg

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        L = random.randint(self.cfg.min_len, self.cfg.seq_len - 1)  # leave room for CLS later
        # Sample tokens in [3, vocab_size-1] to avoid colliding with PAD(0), CLS(1), TARGET(2)
        tokens = [random.randint(3, self.cfg.vocab_size - 1) for _ in range(L)]
        # With prob 0.5, inject the target token somewhere
        label = 0
        if random.random() < 0.5:
            pos = random.randint(0, L - 1)
            tokens[pos] = self.cfg.target_token
            label = 1
        return tokens, label


def collate_batch(batch: List[Tuple[List[int], int]], pad_id: int, cls_id: int) -> Tuple[Tensor, Tensor]:
    """Pads variable-length lists and prepends [CLS].

    Returns:
        x: [B, T] padded token ids (including CLS at index 0)
        y: [B] class labels (0/1)
    """
    sequences, labels = zip(*batch)
    # Prepend CLS to each sequence
    sequences = [[cls_id] + seq for seq in sequences]
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_id] * (max_len - len(s)) for s in sequences]
    x = torch.tensor(padded, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


# -------------------------------
# Lightning Module
# -------------------------------
class LitTransformerClassifier(pl.LightningModule):
    def __init__(self, vocab_size: int, n_classes: int = 2, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, d_ff: int = 256, max_len: int = 256, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.head = CLSClassifier(d_model, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.encoder(x)
        logits = self.head(h)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # AdamW + simple warmup-free cosine schedule for brevity
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.01)
        return optimizer


# -------------------------------
# DataModule to keep I/O neat
# -------------------------------
class ToyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: ToyConfig, batch_size: int = 128, num_workers: int = 0):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        # Split train/val
        self.train_ds = ToySeqDataset(4000, self.cfg)
        self.val_ds = ToySeqDataset(800, self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_batch(b, self.cfg.pad_id, self.cfg.cls_id),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_batch(b, self.cfg.pad_id, self.cfg.cls_id),
        )


# -------------------------------
# CLI entry
# -------------------------------
if __name__ == "__main__":
    import argparse

    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = ToyConfig(vocab_size=args.vocab_size, seq_len=args.seq_len)

    dm = ToyDataModule(cfg, batch_size=args.batch_size, num_workers=args.num_workers)
    model = LitTransformerClassifier(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        pad_id=cfg.pad_id,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        precision=32,
    )

    trainer.fit(model, datamodule=dm)

    # Quick sanity check after training
    batch = next(iter(dm.val_dataloader()))
    x, y = batch
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    print(f"Validation batch accuracy: {acc:.3f}")

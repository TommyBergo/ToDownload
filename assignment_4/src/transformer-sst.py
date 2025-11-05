import os
import argparse
import random
import math
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from gensim.models import Word2Vec

# ---------------------------
# Constants
# ---------------------------
PAD = "<pad>"
UNK = "<unk>"


# ---------------------------
# Tokenization & Vocabulary
# ---------------------------
def tokenize(s: str) -> List[str]:
    """Simple lowercase whitespace tokenization."""
    return s.lower().split()


def build_vocab(texts: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary and frequency-based word-index mappings."""
    freq: Dict[str, int] = {}
    for t in texts:
        for w in tokenize(t):
            freq[w] = freq.get(w, 0) + 1

    vocab = [PAD, UNK] + [w for w, c in freq.items() if c >= min_freq]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


# ---------------------------
# GloVe & Embeddings
# ---------------------------
def _load_glove_generic(path: str) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GloVe file not found: {path}")

    embeddings: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if not parts:
                continue
            word, vec = parts[0], np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings


def load_glove_50d(path: str) -> Dict[str, np.ndarray]:
    """Load 50-dimensional GloVe vectors."""
    emb = _load_glove_generic(path)
    if len(next(iter(emb.values()))) != 50:
        raise ValueError("Expected 50-dimensional GloVe vectors.")
    return emb


def load_glove_300d(path: str) -> Dict[str, np.ndarray]:
    """Load 300-dimensional GloVe vectors."""
    emb = _load_glove_generic(path)
    if len(next(iter(emb.values()))) != 300:
        raise ValueError("Expected 300-dimensional GloVe vectors.")
    return emb


def make_embedding_matrix(w2i: Dict[str, int], glove: Dict[str, np.ndarray]) -> np.ndarray:
    """Construct an embedding matrix from pretrained GloVe embeddings."""
    dim = len(next(iter(glove.values())))
    mat = np.random.normal(0, 0.1, (len(w2i), dim)).astype(np.float32)

    pad_idx = w2i[PAD]
    mat[pad_idx] = 0.0

    for w, idx in w2i.items():
        if w in glove:
            mat[idx] = glove[w]

    return mat


def make_embedding_matrix_from_w2v(w2i: Dict[str, int], w2v_model: Word2Vec) -> np.ndarray:
    """Construct an embedding matrix from a trained Word2Vec model."""
    dim = w2v_model.vector_size
    mat = np.random.normal(0, 0.1, (len(w2i), dim)).astype(np.float32)
    mat[w2i[PAD]] = 0.0

    for w, idx in w2i.items():
        if w in w2v_model.wv:
            mat[idx] = w2v_model.wv[w]

    return mat


# ---------------------------
# Dataset & Collate Function
# ---------------------------
class SST5Dataset(Dataset):
    """SST-5 Dataset compatible with PyTorch DataLoader."""
    def __init__(self, texts: List[str], labels: List[int], w2i: Dict[str, int]):
        self.texts = texts
        self.labels = labels
        self.w2i = w2i

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = [self.w2i.get(w, self.w2i[UNK]) for w in tokenize(self.texts[i])]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)


def build_collate_fn_for_transformer(pad_idx: int):
    """Build collate function with dynamic padding and attention masks."""
    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        X, masks, y = [], [], []
        for ids, lab in batch:
            pad_len = maxlen - len(ids)
            if pad_len > 0:
                pad = torch.full((pad_len,), pad_idx, dtype=torch.long)
                ids = torch.cat([ids, pad])
            mask = (ids != pad_idx).long()
            X.append(ids)
            masks.append(mask)
            y.append(lab)
        return torch.stack(X), torch.stack(masks), torch.stack(y)
    return collate


# ---------------------------
# Model Components
# ---------------------------
class PositionalEncoding(nn.Module):
    """Standard positional encoding used in Transformer models."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer-based sentence classifier."""
    def __init__(
        self, vocab_size: int, emb_matrix: np.ndarray, num_classes: int, pad_idx: int,
        num_layers: int = 2, nhead: int = 4, dim_feedforward: int = 512, dropout: float = 0.1
    ):
        super().__init__()

        emb_dim = emb_matrix.shape[1]
        self.pad_idx = pad_idx

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.emb.weight.requires_grad = True

        # Positional encoder
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.emb(ids)
        x = self.pos_encoder(x)

        src_key_padding_mask = (attention_mask == 0)
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        mask = attention_mask.unsqueeze(-1)
        summed = (encoded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        sent_repr = summed / lengths

        return self.fc(sent_repr)


# ---------------------------
# Training & Evaluation
# ---------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0

    for xb, maskb, yb in loader:
        xb, maskb, yb = xb.to(device), maskb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb, maskb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model performance."""
    model.eval()
    preds_all, gold_all = [], []

    for xb, maskb, yb in loader:
        xb, maskb, yb = xb.to(device), maskb.to(device), yb.to(device)
        logits = model(xb, maskb)
        preds = logits.argmax(dim=1)

        preds_all.append(preds.cpu())
        gold_all.append(yb.cpu())

    preds = torch.cat(preds_all).numpy()
    gold = torch.cat(gold_all).numpy()

    f1 = f1_score(gold, preds, average="macro")
    acc = (preds == gold).mean()
    return acc, f1, preds, gold


# ---------------------------
# Main Script
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove50", required=True)
    parser.add_argument("--glove300", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    # ---------------------------
    # Reproducibility
    # ---------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------------
    # Data Loading
    # ---------------------------
    print("\nLoading SST-5 dataset...")
    sst = load_dataset("SetFit/sst5")
    tr_text, tr_y = sst["train"]["text"], sst["train"]["label"]
    te_text, te_y = sst["test"]["text"], sst["test"]["label"]

    # Vocabulary
    w2i, i2w = build_vocab(tr_text, min_freq=args.min_freq)
    pad_idx = w2i[PAD]
    collate = build_collate_fn_for_transformer(pad_idx)

    tr_loader = DataLoader(SST5Dataset(tr_text, tr_y, w2i), batch_size=args.bs, shuffle=True, collate_fn=collate)
    te_loader = DataLoader(SST5Dataset(te_text, te_y, w2i), batch_size=args.bs, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class weighting
    counts = torch.bincount(torch.tensor(tr_y), minlength=5).float()
    class_w = (counts.sum() / (5 * counts))
    class_w[torch.isinf(class_w)] = 0
    class_w = class_w.to(device)

    # ---------------------------
    # Embeddings
    # ---------------------------
    print(f"\nLoading pretrained embeddings:")
    print(f" - GloVe 50D: {args.glove50}")
    glove50 = load_glove_50d(args.glove50)

    print(f" - GloVe 300D: {args.glove300}")
    glove300 = load_glove_300d(args.glove300)

    print("\nTraining Word2Vec 300D (Skip-gram)...")
    tokens = [tokenize(t) for t in tr_text]
    w2v_300 = Word2Vec(sentences=tokens, vector_size=300, window=5, min_count=2, sg=1, workers=4, epochs=10)

    emb_glove50 = make_embedding_matrix(w2i, glove50)
    emb_glove300 = make_embedding_matrix(w2i, glove300)
    emb_w2v300 = make_embedding_matrix_from_w2v(w2i, w2v_300)

    # ---------------------------
    # Training Loop
    # ---------------------------
    results = []
    embedding_sets = [
        ("GloVe-50D", emb_glove50),
        ("GloVe-300D", emb_glove300),
        ("Word2Vec-300D", emb_w2v300)
    ]

    for name, emb in embedding_sets:
        print(f"\n===== Training Transformer with {name} =====")

        model = TransformerClassifier(
            vocab_size=len(w2i),
            emb_matrix=emb,
            num_classes=5,
            pad_idx=pad_idx,
            num_layers=2,
            nhead=5,
            dim_feedforward=512,
            dropout=0.1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(weight=class_w)

        best_f1, patience_counter = 0.0, 0

        for ep in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, tr_loader, optimizer, criterion, device)
            acc, f1, _, _ = evaluate(model, te_loader, device)
            print(f"Epoch {ep:02d}/{args.epochs} | Train Loss: {tr_loss:.4f} | Test F1: {f1:.4f}")

            # Early stopping
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered after {ep} epochs (no improvement for {args.patience} epochs).")
                    break

        # Final evaluation
        acc, f1, preds, gold = evaluate(model, te_loader, device)
        print(f"\n=== Final Test Results ({name}) ===")
        print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
        print(classification_report(
            gold, preds,
            labels=[0, 1, 2, 3, 4],
            target_names=["very negative", "negative", "neutral", "positive", "very positive"],
            digits=4
        ))

        results.append((name, acc, f1))

    # ---------------------------
    # Summary
    # ---------------------------
    print("\n===== Summary (Test Only) =====")
    for name, acc, f1 in results:
        print(f"Transformer + {name:15s} | Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")


if __name__ == "__main__":
    main()

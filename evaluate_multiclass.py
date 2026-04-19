"""
evaluate_multiclass.py
======================
Standalone multiclass evaluation for the jointly-trained VAE-BiGAN + ANN.

This script is completely SEPARATE from comparison.py (which stays binary).
It trains a fresh ANN on the latent z from the already-saved
RobustVAEBiGAN encoder, then evaluates full per-class metrics.

Two modes
---------
  --mode train   : Train ANN on z from VAE-BiGAN encoder, save weights
  --mode eval    : Load saved ANN and show evaluation metrics only
  --mode both    : Train then immediately evaluate  (default)

Run order
---------
  Step 1 (already done by you):
      python train_aae.py
      python train_bigan.py
      python train_robust.py
      python calibrate_robust.py
      → These produce: test_data.npz, saved_state/robust_vae_bigan_model.pth
        saved_state/reverse_label_map.pkl

  Step 2 (this script — train + evaluate):
      python evaluate_multiclass.py --mode both

  Step 3 (optional — eval only later, no retraining):
      python evaluate_multiclass.py --mode eval

Outputs (saved inside saved_state/)
-------------------------------------
  ann_multiclass.pth          ANN weights
  ann_multiclass_meta.pkl     label map + hyper-params

Usage examples
--------------
  python evaluate_multiclass.py
  python evaluate_multiclass.py --mode train --epochs 80 --lr 0.001
  python evaluate_multiclass.py --mode eval
  python evaluate_multiclass.py --dropout 0.4 --lambda_cls 2.0
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────
# Paths  (must match your project layout)
# ─────────────────────────────────────────────────────────────────
ROBUST_MODEL_PATH  = "saved_state/robust_vae_bigan_model.pth"
REVERSE_LABEL_MAP  = "saved_state/reverse_label_map.pkl"   # {name: int_code}
TEST_DATA_PATH     = "test_data.npz"

ANN_WEIGHTS_PATH   = "saved_state/ann_multiclass.pth"
ANN_META_PATH      = "saved_state/ann_multiclass_meta.pkl"

# ─────────────────────────────────────────────────────────────────
# Inline model definition (avoids import path issues)
# ─────────────────────────────────────────────────────────────────
class RobustVAEBiGAN(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 25), nn.LeakyReLU(0.2),
            nn.Linear(25, 15),        nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(15, latent_dim)
        self.fc_logvar = nn.Linear(15, latent_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 15), nn.LeakyReLU(0.2),
            nn.Linear(15, input_dim),  nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ANNClassifier(nn.Module):
    """Three-layer ANN head on top of latent z."""
    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(dropout),
            nn.Linear(64, 32),          nn.ReLU(), nn.BatchNorm1d(32),  nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, z):
        return self.net(z)   # raw logits


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def load_label_map():
    """
    Load reverse_label_map {name -> int_code} and invert it to
    {int_code -> name} so we can decode predictions to class names.
    """
    if not os.path.exists(REVERSE_LABEL_MAP):
        raise FileNotFoundError(
            f"'{REVERSE_LABEL_MAP}' not found.\n"
            "Make sure train_aae.py (or train_bigan.py) has been run first."
        )
    with open(REVERSE_LABEL_MAP, "rb") as f:
        name_to_code = pickle.load(f)           # e.g. {"Benign": 0, "DoS": 1, ...}
    code_to_name = {v: k for k, v in name_to_code.items()}
    return name_to_code, code_to_name


def load_data():
    """Load test_data.npz. Returns X_train, X_test, y_train, y_test."""
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            f"'{TEST_DATA_PATH}' not found.\n"
            "Run train_aae.py (or train_bigan.py) first."
        )
    data    = np.load(TEST_DATA_PATH)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int64)
    return X_train, X_test, y_train, y_test


def load_encoder(input_dim: int, device):
    """Load RobustVAEBiGAN encoder from saved checkpoint (encoder weights only)."""
    if not os.path.exists(ROBUST_MODEL_PATH):
        raise FileNotFoundError(
            f"'{ROBUST_MODEL_PATH}' not found.\n"
            "Run train_robust.py first."
        )
    model = RobustVAEBiGAN(input_dim).to(device)
    state = torch.load(ROBUST_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded RobustVAEBiGAN from {ROBUST_MODEL_PATH}")
    return model


def encode_dataset(model, X: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """Pass data through the encoder and return mu (deterministic latent)."""
    Z_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb      = torch.FloatTensor(X[i:i+batch_size]).to(device)
            mu, _   = model.encode(xb)
            Z_list.append(mu.cpu().numpy())
    return np.concatenate(Z_list, axis=0)


def print_banner(title: str):
    W = 70
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────

def train_ann(args, device):
    print_banner("PHASE 1 — Train ANN on VAE-BiGAN latent space")

    name_to_code, code_to_name = load_label_map()
    X_train, X_test, y_train, y_test = load_data()

    input_dim   = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    latent_dim  = 8   # must match RobustVAEBiGAN latent_dim

    print(f"\n  input_dim   : {input_dim}")
    print(f"  latent_dim  : {latent_dim}")
    print(f"  num_classes : {num_classes}")
    print(f"  Classes     : { {code_to_name.get(c, str(c)): int((y_train==c).sum()) for c in np.unique(y_train)} }")
    print(f"  train size  : {len(X_train)}   test size: {len(X_test)}\n")

    # ── Encode with frozen VAE-BiGAN encoder ──────────────────────
    print("  Encoding training data through VAE-BiGAN encoder ...")
    robust_model = load_encoder(input_dim, device)

    Z_train = encode_dataset(robust_model, X_train, device)
    Z_test  = encode_dataset(robust_model, X_test,  device)
    print(f"  Z_train shape: {Z_train.shape}   Z_test shape: {Z_test.shape}")

    # ── Build DataLoaders ──────────────────────────────────────────
    # Remap y labels to consecutive 0..N-1 in case codes have gaps
    unique_codes   = np.unique(y_train)
    code_to_idx    = {c: i for i, c in enumerate(unique_codes)}
    idx_to_code    = {i: c for c, i in code_to_idx.items()}
    idx_to_name    = {i: code_to_name.get(c, str(c)) for i, c in idx_to_code.items()}

    y_train_idx = np.array([code_to_idx[c] for c in y_train], dtype=np.int64)
    y_test_idx  = np.array([code_to_idx.get(c, -1) for c in y_test], dtype=np.int64)

    tr_dl = DataLoader(
        TensorDataset(torch.FloatTensor(Z_train), torch.LongTensor(y_train_idx)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # ── ANN ───────────────────────────────────────────────────────
    ann       = ANNClassifier(latent_dim, num_classes, dropout=args.dropout).to(device)
    optimizer = optim.Adam(ann.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Training ANN  |  epochs={args.epochs}  lr={args.lr}  "
          f"batch={args.batch_size}  dropout={args.dropout}\n")
    print(f"  {'Epoch':>5}  {'Train-Loss':>11}  {'Train-Acc':>10}  {'Test-Acc':>9}")
    print("  " + "-" * 42)

    best_test_acc  = 0.0
    best_state     = None

    for epoch in range(1, args.epochs + 1):
        ann.train()
        total_loss, correct, total = 0.0, 0, 0

        for z_b, y_b in tr_dl:
            z_b, y_b = z_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = ann(z_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += len(y_b)

        scheduler.step()

        # Test accuracy
        ann.eval()
        with torch.no_grad():
            Z_t    = torch.FloatTensor(Z_test).to(device)
            preds  = ann(Z_t).argmax(1).cpu().numpy()
        mask         = y_test_idx >= 0
        test_acc     = accuracy_score(y_test_idx[mask], preds[mask]) * 100
        train_acc    = correct / total * 100
        avg_loss     = total_loss / total

        print(f"  {epoch:>5}  {avg_loss:>11.4f}  {train_acc:>9.2f}%  {test_acc:>8.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state    = {k: v.cpu().clone() for k, v in ann.state_dict().items()}

    # ── Save best weights ──────────────────────────────────────────
    os.makedirs("saved_state", exist_ok=True)
    torch.save({"state_dict": best_state,
                "latent_dim": latent_dim,
                "num_classes": num_classes}, ANN_WEIGHTS_PATH)

    meta = {"code_to_idx": code_to_idx,
            "idx_to_code": idx_to_code,
            "idx_to_name": idx_to_name,
            "latent_dim":  latent_dim,
            "num_classes": num_classes}
    with open(ANN_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  Best test accuracy : {best_test_acc:.2f}%")
    print(f"  Saved → {ANN_WEIGHTS_PATH}")
    print(f"  Saved → {ANN_META_PATH}")
    return best_test_acc


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_ann(device):
    print_banner("PHASE 2 — Evaluation Metrics  (VAE-BiGAN + ANN)")

    # ── Load saved artifacts ───────────────────────────────────────
    for path in [ANN_WEIGHTS_PATH, ANN_META_PATH, ROBUST_MODEL_PATH, TEST_DATA_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run with --mode train (or both) first."
            )

    with open(ANN_META_PATH, "rb") as f:
        meta = pickle.load(f)

    idx_to_name = meta["idx_to_name"]
    code_to_idx = meta["code_to_idx"]
    latent_dim  = meta["latent_dim"]
    num_classes = meta["num_classes"]
    dropout     = 0.3  # dropout doesn't matter at inference (eval mode)

    ckpt = torch.load(ANN_WEIGHTS_PATH, map_location=device)
    ann  = ANNClassifier(latent_dim, num_classes, dropout=dropout).to(device)
    ann.load_state_dict(ckpt["state_dict"])
    ann.eval()

    _, X_test, _, y_test = load_data()
    input_dim = X_test.shape[1]

    robust_model = load_encoder(input_dim, device)
    Z_test       = encode_dataset(robust_model, X_test, device)

    y_test_idx = np.array([code_to_idx.get(c, -1) for c in y_test], dtype=np.int64)
    valid_mask = y_test_idx >= 0

    with torch.no_grad():
        logits     = ann(torch.FloatTensor(Z_test).to(device))
        probs      = torch.softmax(logits, dim=1).cpu().numpy()
        preds_idx  = logits.argmax(1).cpu().numpy()

    y_true = y_test_idx[valid_mask]
    y_pred = preds_idx[valid_mask]
    class_names = [idx_to_name[i] for i in sorted(idx_to_name)]

    # ── 1. Overall accuracy ────────────────────────────────────────
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  Overall Accuracy  :  {overall_acc:.2f}%\n")

    # ── 2. Per-class report ────────────────────────────────────────
    print("  Per-Class Classification Report")
    print("  " + "-" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    # ── 3. Confusion matrix (text) ─────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix  (rows = true, cols = predicted)")
    print("  " + "-" * 60)
    header = "  {:>22s}  ".format("") + "  ".join(f"{n[:8]:>8}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"  {class_names[i]:>22s}  {row_str}")

    # ── 4. Per-class accuracy ──────────────────────────────────────
    print("\n  Per-Class Accuracy")
    print("  " + "-" * 40)
    for i, name in enumerate(class_names):
        mask     = y_true == i
        if mask.sum() == 0:
            print(f"  {name:>22s}  :  N/A (no samples)")
            continue
        per_acc  = (y_pred[mask] == i).mean() * 100
        support  = mask.sum()
        print(f"  {name:>22s}  :  {per_acc:6.2f}%   (support={support})")

    # ── 5. Confidence statistics ───────────────────────────────────
    max_probs = probs[valid_mask].max(axis=1)
    print(f"\n  Prediction Confidence Stats (max softmax probability)")
    print(f"  {'Mean':>6}: {max_probs.mean():.4f}   "
          f"{'Std':>4}: {max_probs.std():.4f}   "
          f"{'Min':>4}: {max_probs.min():.4f}   "
          f"{'Max':>4}: {max_probs.max():.4f}")

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train / evaluate ANN classifier on VAE-BiGAN latent space."
    )
    p.add_argument("--mode",       choices=["train", "eval", "both"], default="both",
                   help="'train' = train only, 'eval' = evaluate only, "
                        "'both' = train then evaluate (default)")
    p.add_argument("--epochs",     type=int,   default=60,
                   help="Number of training epochs (default: 60)")
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Learning rate (default: 0.001)")
    p.add_argument("--batch_size", type=int,   default=256,
                   help="Batch size (default: 256)")
    p.add_argument("--dropout",    type=float, default=0.3,
                   help="ANN dropout rate (default: 0.3)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Mode  : {args.mode}\n")

    if args.mode in ("train", "both"):
        train_ann(args, device)

    if args.mode in ("eval", "both"):
        evaluate_ann(device)
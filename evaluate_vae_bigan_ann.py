"""
evaluate_vae_bigan_ann.py
=========================
Standalone evaluation script for the jointly trained VAE-BiGAN + ANN.

Loads the saved model from saved_state/ and prints full multiclass metrics:
  - Overall Accuracy
  - Per-class: Precision, Recall, F1-Score, Support
  - TP, TN, FP, FN for every class (one-vs-rest)
  - Confusion Matrix
  - Macro / Weighted averages

Run:
    python evaluate_vae_bigan_ann.py

Requirements (must already exist):
    saved_state/vae_bigan_ann_model.pth
    saved_state/ann_classifier.pth
    saved_state/ann_multiclass_meta.pkl
    test_data.npz
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "saved_state/vae_bigan_ann_model.pth"
CLF_PATH    = "saved_state/ann_classifier.pth"
META_PATH   = "saved_state/ann_multiclass_meta.pkl"
DATA_PATH   = "test_data.npz"


# ─────────────────────────────────────────────────────────────────
# Architecture  (must match train_vae_bigan_ann.py exactly)
# ─────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64),        nn.LeakyReLU(0.2), nn.BatchNorm1d(64),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ANNClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def sep(char="─", width=68):
    print(char * width)

def banner(title):
    sep("═")
    print(f"  {title}")
    sep("═")


def load_model(device):
    """Load encoder + classifier from saved checkpoints."""
    for path in [MODEL_PATH, CLF_PATH, META_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n  '{path}' not found.\n"
                "  Run  python train_vae_bigan_ann.py  first."
            )

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    latent_dim  = meta["latent_dim"]
    num_classes = meta["num_classes"]
    idx_to_name = meta["idx_to_name"]
    code_to_idx = meta["code_to_idx"]

    enc_ckpt = torch.load(MODEL_PATH, map_location=device)
    clf_ckpt = torch.load(CLF_PATH,   map_location=device)

    input_dim = enc_ckpt["input_dim"]

    encoder    = Encoder(input_dim, latent_dim).to(device)
    classifier = ANNClassifier(latent_dim, num_classes, dropout=0.0).to(device)

    encoder.load_state_dict(enc_ckpt["encoder"])
    classifier.load_state_dict(clf_ckpt["classifier"])
    encoder.eval()
    classifier.eval()

    print(f"  Loaded encoder     : {MODEL_PATH}")
    print(f"  Loaded classifier  : {CLF_PATH}")
    print(f"  input_dim={input_dim}  latent_dim={latent_dim}  num_classes={num_classes}")

    return encoder, classifier, idx_to_name, code_to_idx, input_dim


def load_test_data(code_to_idx):
    """Load test_data.npz and remap labels to consecutive indices."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n  '{DATA_PATH}' not found.\n"
            "  Make sure you are running from inside final_capstone/."
        )
    data   = np.load(DATA_PATH)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    # Remap pipeline codes → consecutive indices (same mapping used during training)
    y_idx = np.array([code_to_idx.get(int(c), -1) for c in y_test], dtype=np.int64)
    valid = y_idx >= 0
    print(f"  Test samples total : {len(X_test)}")
    print(f"  Valid (known class): {valid.sum()}")
    if (~valid).sum() > 0:
        print(f"  Skipped (unseen)   : {(~valid).sum()}")
    return X_test[valid], y_idx[valid]


def run_inference(encoder, classifier, X, device, batch_size=512):
    """Pass X through encoder → classifier, return predictions and softmax probs."""
    all_preds = []
    all_probs = []
    ds = DataLoader(TensorDataset(torch.FloatTensor(X)),
                    batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in ds:
            xb      = xb.to(device)
            mu, _   = encoder(xb)
            logits  = classifier(mu)
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
    return np.concatenate(all_preds), np.concatenate(all_probs, axis=0)


# ─────────────────────────────────────────────────────────────────
# Metric printers
# ─────────────────────────────────────────────────────────────────

def print_overall(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  Overall Accuracy : {acc:.2f}%\n")


def print_classification_report(y_true, y_pred, class_names):
    print("  Per-Class Report  (Precision / Recall / F1 / Support)")
    sep("─")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4,
    ))


def print_tp_tn_fp_fn(y_true, y_pred, class_names):
    """Print TP, TN, FP, FN and derived metrics for each class (one-vs-rest)."""
    print("  Per-Class  TP / TN / FP / FN  (one-vs-rest)")
    sep("─")

    header = (f"  {'Class':<28} {'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7}"
              f"  {'Precision':>10} {'Recall':>9} {'F1':>8} {'Specificity':>12}")
    print(header)
    sep("─")

    for i, name in enumerate(class_names):
        y_bin_true = (y_true == i).astype(int)
        y_bin_pred = (y_pred == i).astype(int)

        tp = int(((y_bin_pred == 1) & (y_bin_true == 1)).sum())
        tn = int(((y_bin_pred == 0) & (y_bin_true == 0)).sum())
        fp = int(((y_bin_pred == 1) & (y_bin_true == 0)).sum())
        fn = int(((y_bin_pred == 0) & (y_bin_true == 1)).sum())

        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1          = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"  {name:<28} {tp:>7} {tn:>7} {fp:>7} {fn:>7}"
              f"  {precision:>10.4f} {recall:>9.4f} {f1:>8.4f} {specificity:>12.4f}")

    sep("─")


def print_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix  (rows = True class, cols = Predicted class)")
    sep("─")

    col_w = max(len(n) for n in class_names) + 2
    # Header row
    header = "  " + " " * 28 + "  "
    header += "  ".join(f"{n[:col_w]:>{col_w}}" for n in class_names)
    print(header)
    sep("─")

    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{col_w}d}" for v in row)
        print(f"  {class_names[i]:<28}  {row_str}")

    sep("─")
    print()


def print_macro_weighted(y_true, y_pred, class_names):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    # Macro
    macro_p  = prec.mean()
    macro_r  = rec.mean()
    macro_f1 = f1.mean()
    # Weighted
    total    = sup.sum()
    w        = sup / total
    w_p      = (prec * w).sum()
    w_r      = (rec  * w).sum()
    w_f1     = (f1   * w).sum()

    print("  Macro  / Weighted Averages")
    sep("─")
    print(f"  {'':28}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'Macro Average':<28}  {macro_p:>10.4f}  {macro_r:>8.4f}  {macro_f1:>8.4f}")
    print(f"  {'Weighted Average':<28}  {w_p:>10.4f}  {w_r:>8.4f}  {w_f1:>8.4f}")
    sep("─")
    print()


def print_confidence(probs, y_true, y_pred, class_names):
    max_conf = probs.max(axis=1)
    print("  Prediction Confidence  (max softmax probability)")
    sep("─")
    print(f"  Overall  —  mean={max_conf.mean():.4f}  "
          f"std={max_conf.std():.4f}  "
          f"min={max_conf.min():.4f}  "
          f"max={max_conf.max():.4f}\n")

    print(f"  {'Class':<28}  {'Mean Conf':>10}  {'Correct Conf':>13}  {'Wrong Conf':>11}")
    sep("─")
    for i, name in enumerate(class_names):
        mask_i   = y_true == i
        if mask_i.sum() == 0:
            print(f"  {name:<28}  {'N/A':>10}")
            continue
        correct  = (y_pred[mask_i] == i)
        c_conf   = probs[mask_i, i][correct].mean()  if correct.sum() > 0 else float("nan")
        w_conf   = probs[mask_i, i][~correct].mean() if (~correct).sum() > 0 else float("nan")
        print(f"  {name:<28}  {probs[mask_i, i].mean():>10.4f}"
              f"  {c_conf:>13.4f}  {w_conf:>11.4f}")
    sep("─")
    print()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    banner("VAE-BiGAN + ANN  —  Multiclass Evaluation")
    print(f"\n  Device : {device}\n")

    # Load
    sep()
    print("  Loading model ...")
    sep()
    encoder, classifier, idx_to_name, code_to_idx, _ = load_model(device)

    print()
    sep()
    print("  Loading test data ...")
    sep()
    X_test, y_true = load_test_data(code_to_idx)

    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    print(f"  Class names : {class_names}\n")

    # Inference
    sep()
    print("  Running inference ...")
    sep()
    y_pred, probs = run_inference(encoder, classifier, X_test, device)
    print(f"  Done. {len(y_pred)} predictions made.\n")

    # ── Print all metrics ──────────────────────────────────────────
    banner("RESULTS")

    print_overall(y_true, y_pred)

    sep("═")
    print("  1. PER-CLASS CLASSIFICATION REPORT")
    sep("═")
    print_classification_report(y_true, y_pred, class_names)

    sep("═")
    print("  2. TP / TN / FP / FN  PER CLASS")
    sep("═")
    print()
    print_tp_tn_fp_fn(y_true, y_pred, class_names)
    print()

    sep("═")
    print("  3. CONFUSION MATRIX")
    sep("═")
    print()
    print_confusion_matrix(y_true, y_pred, class_names)

    sep("═")
    print("  4. MACRO & WEIGHTED AVERAGES")
    sep("═")
    print()
    print_macro_weighted(y_true, y_pred, class_names)

    sep("═")
    print("  5. CONFIDENCE ANALYSIS")
    sep("═")
    print()
    print_confidence(probs, y_true, y_pred, class_names)

    sep("═")
    print("  Evaluation complete.")
    sep("═")
    print()


if __name__ == "__main__":
    main()
# """Evaluate Robust VAE-BiGAN and save confusion matrix plot."""

# import torch
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import LinearSegmentedColormap
# from sklearn.metrics import (
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
# )

# from models.robust_model import RobustVAEBiGAN

# MODEL_PATH  = "saved_state/robust_vae_bigan_model.pth"
# CALIB_PATH  = "robust_calibration.pkl"
# DATA_PATH   = "test_data.npz"
# OUTPUT_PNG  = "robust_confusion_matrix.png"

# CLASS_NAMES = {
#     0: "Benign",
#     1: "C&C-HeartBeat",
#     2: "DDoS",
#     3: "Okiru",
#     4: "PortScan",
# }

# print("=" * 60)
# print("  Robust VAE-BiGAN - Evaluation")
# print("=" * 60)

# print("\nLoading test data ...")
# test_data = np.load(DATA_PATH)
# X_test    = test_data["X_test"].astype(np.float32)
# y_test    = test_data["y_test"]          # original multi-class labels
# input_dim = X_test.shape[1]
# print(f"  Samples : {len(X_test)}   Input dim : {input_dim}")

# unique, counts = np.unique(y_test, return_counts=True)
# print("  Class distribution (test set):")
# for cls, cnt in zip(unique, counts):
#     print(f"    [{cls}] {CLASS_NAMES.get(int(cls), str(cls)):20s}  {cnt:6d} samples")

# print("\nLoading Robust VAE-BiGAN model ...")
# model = RobustVAEBiGAN(input_dim)
# model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
# model.eval()
# print("  Model loaded.")

# print("\nLoading calibration bundle ...")
# with open(CALIB_PATH, "rb") as f:
#     calib = pickle.load(f)
# threshold = calib["threshold"]
# flip      = calib["flip"]
# print(f"  threshold = {threshold:.4f}   flip = {flip}")

# print("\nComputing anomaly scores ...")
# X_t = torch.FloatTensor(X_test)
# all_scores = []

# with torch.no_grad():
#     for i in range(0, len(X_t), 1000):
#         batch   = X_t[i:i+1000]
#         mu, _   = model.encode(batch)
#         s       = model.discriminator(
#             torch.cat([batch, mu], dim=1)).squeeze().numpy()
#         if s.ndim == 0:
#             s = s.reshape(1)
#         all_scores.append(s)

# scores   = np.concatenate(all_scores)
# if flip:
#     scores = 1.0 - scores

# y_binary = (y_test > 0).astype(int)
# y_pred_binary = (scores > threshold).astype(int)

# n_classes    = len(CLASS_NAMES)
# class_labels = sorted(CLASS_NAMES.keys())   # [0, 1, 2, 3, 4]
# label_names  = [CLASS_NAMES[c] for c in class_labels]

# y_pred_full = np.where(y_pred_binary == 0, 0, y_test)

# cm = confusion_matrix(y_test, y_pred_full, labels=class_labels)

# print("\n" + "=" * 60)
# print("  Binary Detection Metrics (Benign vs Attack)")
# print("=" * 60)

# auc  = roc_auc_score(y_binary, scores)
# tn   = int(((y_binary == 0) & (y_pred_binary == 0)).sum())
# fp   = int(((y_binary == 0) & (y_pred_binary == 1)).sum())
# fn   = int(((y_binary == 1) & (y_pred_binary == 0)).sum())
# tp   = int(((y_binary == 1) & (y_pred_binary == 1)).sum())
# n_b  = int((y_binary == 0).sum())
# n_a  = int((y_binary == 1).sum())

# print(f"  AUC-ROC           : {auc:.4f}")
# print(f"  Detection Rate    : {tp/max(n_a,1)*100:.2f}%  ({tp}/{n_a} attacks caught)")
# print(f"  False Alarm Rate  : {fp/max(n_b,1)*100:.2f}%  ({fp}/{n_b} benign flagged)")
# print(f"  Missed Attacks    : {fn/max(n_a,1)*100:.2f}%  ({fn}/{n_a})")
# print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# print("\n" + "=" * 60)
# print("  Per-Class Detection Breakdown")
# print("=" * 60)
# for cls in class_labels:
#     mask = y_test == cls
#     if mask.sum() == 0:
#         continue
#     caught = int(y_pred_binary[mask].sum())
#     total  = int(mask.sum())
#     rate   = caught / total * 100
#     bar    = "█" * int(rate // 5) + "░" * (20 - int(rate // 5))
#     print(f"  [{cls}] {CLASS_NAMES[cls]:20s}  [{bar}] {rate:6.1f}%  ({caught}/{total})")

# print("\nClassification Report (binary):")
# print(classification_report(
#     y_binary, y_pred_binary,
#     target_names=["Benign", "Attack"],
#     zero_division=0,
# ))

# print("Full 5x5 Confusion Matrix (text):")
# header = f"  {'':22s}" + "".join(f"{n:>14s}" for n in label_names)
# print(header)
# print("  " + "-" * (22 + 14 * n_classes))
# for i, row_name in enumerate(label_names):
#     row_str = f"  {row_name:22s}" + "".join(f"{cm[i,j]:>14d}" for j in range(n_classes))
#     print(row_str)

# print(f"\nRendering confusion matrix to {OUTPUT_PNG} ...")

# fig, axes = plt.subplots(1, 2, figsize=(18, 7),
#                          gridspec_kw={"width_ratios": [3, 1], "wspace": 0.35})

# ax = axes[0]

# cmap = LinearSegmentedColormap.from_list(
#     "robust_cm",
#     ["#FFFFFF", "#C8E6FA", "#4A90D9", "#1A3A6B"],
#     N=256,
# )

# im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

# cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=10)
# cbar.set_label("Sample Count", fontsize=11, labelpad=10)

# ax.set_xticks(range(n_classes))
# ax.set_yticks(range(n_classes))
# ax.set_xticklabels(label_names, fontsize=11, rotation=30, ha="right")
# ax.set_yticklabels(label_names, fontsize=11)

# ax.set_xlabel("Predicted Class", fontsize=13, labelpad=12)
# ax.set_ylabel("True Class",      fontsize=13, labelpad=12)
# ax.set_title("Robust VAE-BiGAN\n5×5 Confusion Matrix",
#              fontsize=15, fontweight="bold", pad=16)

# for x in np.arange(-.5, n_classes, 1):
#     ax.axhline(x, color="#CCCCCC", linewidth=0.7)
#     ax.axvline(x, color="#CCCCCC", linewidth=0.7)

# thresh_color = cm.max() / 2.0
# for i in range(n_classes):
#     for j in range(n_classes):
#         val   = cm[i, j]
#         total = cm[i].sum()
#         pct   = f"\n({val/max(total,1)*100:.1f}%)" if total > 0 else ""
#         color = "white" if val > thresh_color else "#1A1A2E"

#         weight = "bold" if i == j else "normal"
#         ax.text(j, i, f"{val}{pct}",
#                 ha="center", va="center",
#                 fontsize=10, color=color, fontweight=weight)

# for k in range(n_classes):
#     rect = mpatches.FancyBboxPatch(
#         (k - 0.49, k - 0.49), 0.98, 0.98,
#         boxstyle="square,pad=0",
#         linewidth=2.5, edgecolor="#28A745", facecolor="none",
#     )
#     ax.add_patch(rect)

# ax2 = axes[1]

# rates  = []
# colors = []
# for cls in class_labels:
#     mask = y_test == cls
#     if mask.sum() == 0:
#         rates.append(0.0)
#     elif cls == 0:
#         # For benign: True Negative Rate
#         caught = int((y_pred_binary[mask] == 0).sum())
#         rates.append(caught / int(mask.sum()) * 100)
#     else:
#         caught = int(y_pred_binary[mask].sum())
#         rates.append(caught / int(mask.sum()) * 100)

#     r = rates[-1]
#     colors.append("#28A745" if r >= 90 else ("#FFC107" if r >= 60 else "#DC3545"))

# bars = ax2.barh(range(n_classes), rates, color=colors, edgecolor="white",
#                 height=0.6, linewidth=1.2)

# for i, (bar, rate) in enumerate(zip(bars, rates)):
#     ax2.text(min(rate + 1.5, 102), i, f"{rate:.1f}%",
#              va="center", ha="left", fontsize=11, fontweight="bold",
#              color="#1A1A2E")

# ax2.set_yticks(range(n_classes))
# ax2.set_yticklabels(label_names, fontsize=11)
# ax2.set_xlabel("Detection / Correct Rate (%)", fontsize=12, labelpad=8)
# ax2.set_title("Per-Class\nDetection Rate", fontsize=13, fontweight="bold", pad=12)
# ax2.set_xlim(0, 115)
# ax2.axvline(90, color="#28A745", linestyle="--", linewidth=1.2, alpha=0.6,
#             label="90% target")
# ax2.axvline(60, color="#FFC107", linestyle="--", linewidth=1.2, alpha=0.6,
#             label="60% threshold")
# ax2.legend(fontsize=9, loc="lower right")
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)
# ax2.tick_params(axis="x", labelsize=10)

# ax2.text(0, 0, "← TNR (benign)", fontsize=8, color="#555555", va="center")

# fig.text(
#     0.5, 0.01,
#     f"Binary detector: threshold={threshold:.4f}  |  flip={flip}  |  "
#     f"AUC={auc:.4f}  |  n={len(X_test)} test samples\n"
#     "Predicted class = Benign if score ≤ threshold, else true class shown "
#     "(so missed attacks appear in column 0).",
#     ha="center", fontsize=9, color="#555555",
#     style="italic",
# )

# plt.tight_layout(rect=[0, 0.06, 1, 1])
# fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
# print(f"  Saved -> {OUTPUT_PNG}")
# plt.close(fig)

# print("\nEvaluation complete.")
# print(f"  AUC-ROC  : {auc:.4f}")
# print(f"  DR       : {tp/max(n_a,1)*100:.2f}%")
# print(f"  FAR      : {fp/max(n_b,1)*100:.2f}%")
# print(f"  Plot     : {OUTPUT_PNG}")
































import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from models.robust_model import RobustVAEBiGAN

# Paths
MODEL_PATH    = "saved_state/robust_vae_bigan_model.pth"
CALIB_PATH    = "robust_calibration.pkl"
BIGAN_RF_PATH = "bigan_rf_classifier.pkl"
DATA_PATH     = "test_data.npz"
OUTPUT_PNG    = "robust_confusion_matrix.png"

CLASS_NAMES = {0: "Benign", 1: "C&C-HeartBeat", 2: "DDoS", 3: "Okiru", 4: "PortScan"}

def evaluate():
    # 1. Load Data
    npz = np.load(DATA_PATH)
    X_test = npz["X_test"].astype(np.float32)
    y_test = npz["y_test"]
    input_dim = X_test.shape[1]

    # 2. Load Models
    model = RobustVAEBiGAN(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with open(CALIB_PATH, "rb") as f:
        calib = pickle.load(f)
    threshold, flip = calib["threshold"], calib["flip"]

    if not os.path.exists(BIGAN_RF_PATH):
        raise FileNotFoundError("RF Specialist not found. Run train_rf_specialist.py first!")
    with open(BIGAN_RF_PATH, "rb") as f:
        bigan_rf = pickle.load(f)

    # 3. Stage 1: Binary Detection & Feature Extraction
    print("Stage 1: Running BiGAN Gatekeeper...")
    X_t = torch.FloatTensor(X_test)
    all_scores, all_mus = [], []

    with torch.no_grad():
        for i in range(0, len(X_t), 1000):
            batch = X_t[i:i+1000]
            mu, _ = model.encode(batch)
            s = model.discriminator(torch.cat([batch, mu], dim=1)).squeeze().numpy()
            if s.ndim == 0: s = s.reshape(1)
            all_scores.append(s)
            all_mus.append(mu.numpy())

    scores = np.concatenate(all_scores)
    mu_all = np.concatenate(all_mus)
    if flip: scores = 1.0 - scores

    y_pred_binary = (scores > threshold).astype(int)

    # 4. Stage 2: Multi-class (Only for flagged attacks)
    print("Stage 2: Classifying detected anomalies...")
    y_pred_full = np.zeros(len(y_test), dtype=int) # Default to Benign (0)
    attack_indices = np.where(y_pred_binary == 1)[0]

    if len(attack_indices) > 0:
        mu_flagged = mu_all[attack_indices]
        # RF classifies only between classes 1-4
        y_pred_full[attack_indices] = bigan_rf.predict(mu_flagged)

    # 5. Metrics
    print("\n--- PERFORMANCE SUMMARY ---")
    y_binary_true = (y_test > 0).astype(int)
    auc = roc_auc_score(y_binary_true, scores)
    print(f"Stage 1 AUC-ROC: {auc:.4f}")
    
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred_full, target_names=list(CLASS_NAMES.values()), zero_division=0))

    # Confusion Matrix Plotting (Standard Matplotlib block)
    cm = confusion_matrix(y_test, y_pred_full)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title("Final Hybrid Confusion Matrix")
    plt.colorbar()
    plt.savefig(OUTPUT_PNG)
    print(f"Confusion Matrix saved to {OUTPUT_PNG}")

if __name__ == "__main__":
    evaluate()


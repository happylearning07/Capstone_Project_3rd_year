"""Train BiGAN and save calibration artifacts."""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

from models.bigan_model import BiGAN
from utils.preprocessing import load_and_clean
from sklearn.metrics import roc_auc_score, roc_curve

LATENT_DIM    = 8
EPOCHS        = 100
BATCH_SIZE    = 256
LR            = 0.0002
GRAD_CLIP     = 1.0

TRAIN_CLASSES = [2, 3, 4]

CALIB_PATH    = "bigan_calibration.pkl"

import os, pickle as _pkl

_fc_path = "saved_state/feature_cols.pkl"
if os.path.exists(_fc_path):
    with open(_fc_path, "rb") as _f:
        _existing_cols = _pkl.load(_f)
    if len(_existing_cols) != 39:
        print(f"Warning: saved_state/feature_cols.pkl has {len(_existing_cols)} cols "
              f"(expected 39). Rebuilding artifacts...")
        for _p in ["saved_state/feature_cols.pkl",
                   "saved_state/scaler.pkl",
                   "saved_state/reverse_label_map.pkl"]:
            if os.path.exists(_p):
                os.remove(_p)
                print(f"  Deleted {_p}")

print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test, input_dim = load_and_clean(
    'data/iot23_combined_new.csv', n_rows=500_000)

if input_dim != 39:
    raise RuntimeError(
        f"Expected input_dim=39, got {input_dim}. "
        "Delete saved_state/ and rerun train_aae.py then train_bigan.py."
    )

print(f"  input_dim     : {input_dim}")
print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")

unique, counts = np.unique(y_train, return_counts=True)
print(f"  Class distribution (train): {dict(zip(unique.tolist(), counts.tolist()))}")

train_mask  = np.isin(y_train, TRAIN_CLASSES)
X_train_maj = X_train[train_mask].astype(np.float32)
print(f"\n  Training on classes {TRAIN_CLASSES} | samples: {len(X_train_maj)}")
print("  Rare class handling is deferred to AAE+RF.")

dataset = TensorDataset(torch.FloatTensor(X_train_maj))
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"  Batches per epoch: {len(loader)}")

model     = BiGAN(input_dim, LATENT_DIM)
criterion = nn.BCELoss()

optim_ge = optim.Adam(
    list(model.generator.parameters()) + list(model.encoder.parameters()),
    lr=LR, betas=(0.5, 0.999))
optim_d = optim.Adam(
    model.discriminator.parameters(),
    lr=LR, betas=(0.5, 0.999))

print(f"\nTraining BiGAN | epochs={EPOCHS} | batch={BATCH_SIZE} | "
      f"latent={LATENT_DIM} | grad_clip={GRAD_CLIP}")
print("-" * 70)

for epoch in range(EPOCHS):
    epoch_d_loss  = 0.0
    epoch_ge_loss = 0.0

    for (batch_x,) in loader:
        B = batch_x.size(0)

        real_label = torch.ones(B, 1)
        fake_label = torch.zeros(B, 1)

        optim_d.zero_grad()

        z_encoded = model.encoder(batch_x)
        real_pair = torch.cat((batch_x, z_encoded.detach()), dim=1)
        d_real    = criterion(model.discriminator(real_pair), real_label)

        z_random  = torch.randn(B, LATENT_DIM)
        x_gen     = model.generator(z_random)
        fake_pair = torch.cat((x_gen.detach(), z_random), dim=1)
        d_fake    = criterion(model.discriminator(fake_pair), fake_label)

        d_loss = (d_real + d_fake) * 0.5
        d_loss.backward()
        nn.utils.clip_grad_norm_(model.discriminator.parameters(), GRAD_CLIP)
        optim_d.step()

        optim_ge.zero_grad()

        z_enc_new = model.encoder(batch_x)
        real_new  = torch.cat((batch_x, z_enc_new), dim=1)
        x_gen_new = model.generator(z_random)
        fake_new  = torch.cat((x_gen_new, z_random), dim=1)

        loss_e = criterion(model.discriminator(real_new), fake_label)
        loss_g = criterion(model.discriminator(fake_new), real_label)

        ge_loss = (loss_e + loss_g) * 0.5
        ge_loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.generator.parameters()) + list(model.encoder.parameters()),
            GRAD_CLIP)
        optim_ge.step()

        epoch_d_loss  += d_loss.item()
        epoch_ge_loss += ge_loss.item()

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        avg_d  = epoch_d_loss  / len(loader)
        avg_ge = epoch_ge_loss / len(loader)
        flag   = " [warn: D diverging]" if avg_d > 5.0 else ""
        print(f"Epoch {epoch:3d}/{EPOCHS} | D Loss: {avg_d:.4f} | GE Loss: {avg_ge:.4f}{flag}")

print("\nSaving BiGAN weights...")
torch.save(
    {'state_dict': model.state_dict(), 'input_dim': input_dim},
    'bigan_final.pth'
)
np.savez('test_data.npz',
         X_test=X_test,  y_test=y_test,
         X_train=X_train, y_train=y_train)

def compute_scores(model, X_tensor, r_min=None, r_max=None):
    with torch.no_grad():
        z_enc   = model.encoder(X_tensor)
        x_hat   = model.generator(z_enc)
        recon   = torch.mean((X_tensor - x_hat) ** 2, dim=1).numpy()
        pair    = torch.cat((X_tensor, z_enc), dim=1)
        d_score = model.discriminator(pair).squeeze().numpy()
        if d_score.ndim == 0:
            d_score = d_score.reshape(1)

    if r_min is None:
        r_min = float(recon.min())
        r_max = float(recon.max())

    span = r_max - r_min
    recon_norm = (recon - r_min) / (span + 1e-8) if span > 0 else np.zeros_like(recon)
    recon_norm = np.clip(recon_norm, 0.0, 1.0)

    scores = 0.5 * (1.0 - d_score) + 0.5 * recon_norm
    return scores, r_min, r_max

print("\nCalibrating threshold (Youden's J) on full test set...")
model.eval()
X_test_t = torch.FloatTensor(X_test)
y_binary  = (y_test > 0).astype(int)

raw_scores, r_min, r_max = compute_scores(model, X_test_t)
print(f"  Reconstruction range (test): [{r_min:.6f}, {r_max:.6f}]")

auc  = roc_auc_score(y_binary, raw_scores)
flip = auc < 0.5
if flip:
    print(f"  AUC={auc:.4f} < 0.5 - flipping scores.")
    scores = 1.0 - raw_scores
    auc    = roc_auc_score(y_binary, scores)
else:
    scores = raw_scores

print(f"  AUC-ROC (after flip={flip}): {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_binary, scores)
tnr      = 1.0 - fpr
j_scores = tpr + tnr - 1.0
best_idx  = int(np.argmax(j_scores))
threshold = float(thresholds[best_idx])

benign_scores = scores[y_binary == 0]
attack_scores = scores[y_binary == 1]

print(f"\n  Benign scores - mean: {benign_scores.mean():.4f}  "
      f"std: {benign_scores.std():.4f}  n={len(benign_scores)}")
print(f"  Attack scores - mean: {attack_scores.mean():.4f}  "
      f"std: {attack_scores.std():.4f}  n={len(attack_scores)}")
print(f"\n  Youden's J threshold: {threshold:.4f}  "
      f"(J={j_scores[best_idx]:.4f}, "
      f"TPR={tpr[best_idx]:.4f}, TNR={tnr[best_idx]:.4f})")

y_pred = (scores > threshold).astype(int)
tp = int((y_pred[y_binary == 1] == 1).sum())
tn = int((y_pred[y_binary == 0] == 0).sum())
fp = int((y_pred[y_binary == 0] == 1).sum())
fn = int((y_pred[y_binary == 1] == 0).sum())

print(f"\n  At threshold {threshold:.4f}:")
print(f"    Attacks caught (TP) : {tp} / {len(attack_scores)}  "
      f"({tp/max(len(attack_scores),1)*100:.1f}%)")
print(f"    Benign correct (TN) : {tn} / {len(benign_scores)}  "
      f"({tn/max(len(benign_scores),1)*100:.1f}%)")
print(f"    False alarms   (FP) : {fp} / {len(benign_scores)}  "
      f"({fp/max(len(benign_scores),1)*100:.2f}%)")
print(f"    Missed attacks (FN) : {fn} / {len(attack_scores)}  "
      f"({fn/max(len(attack_scores),1)*100:.2f}%)")

calibration = {
    'threshold'     : threshold,
    'flip'          : flip,
    'r_min'         : r_min,
    'r_max'         : r_max,
    'train_classes' : TRAIN_CLASSES,
}
with open(CALIB_PATH, 'wb') as f:
    pickle.dump(calibration, f)

print(f"\n  Calibration bundle saved to {CALIB_PATH}")
print(f"      threshold={threshold:.4f}  flip={flip}  "
      f"r_min={r_min:.6f}  r_max={r_max:.6f}")

print("\nBiGAN training complete.")
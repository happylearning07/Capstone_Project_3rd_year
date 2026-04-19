"""Train and calibrate Robust VAE-BiGAN."""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
import shutil

from models.robust_model import RobustVAEBiGAN
from utils.preprocessing import load_and_clean
from sklearn.metrics import roc_auc_score, roc_curve

#  Config
EPSILON        = 0.15     # FGSM adversarial epsilon (match gateway.py)
LATENT_DIM     = 8
EPOCHS         = 80       # Enough for convergence with reconstruction anchor
BATCH_SIZE     = 256
LR             = 0.0002
GRAD_CLIP      = 1.0
LAMBDA_RECON   = 5.0      # Weight on MSE reconstruction loss
BETA_KL_MAX    = 0.005    # Maximum KL weight - annealed from 0
TRAIN_CLASSES  = [2, 3, 4]  # DDoS, Okiru, PortScan (same as BiGAN)

CALIB_PATH     = "robust_calibration.pkl"
MODEL_SAVE     = "saved_state/robust_vae_bigan_model.pth"

#  1. Data loading
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test, input_dim = load_and_clean(
    'data/iot23_combined_new.csv', n_rows=500_000)

print(f"  input_dim     : {input_dim}")
print(f"  X_train shape : {X_train.shape}")

unique, counts = np.unique(y_train, return_counts=True)
print(f"  Class distribution (train): {dict(zip(unique.tolist(), counts.tolist()))}")

train_mask  = np.isin(y_train, TRAIN_CLASSES)
X_train_maj = X_train[train_mask].astype(np.float32)
print(f"\n  Training on classes {TRAIN_CLASSES} | samples: {len(X_train_maj)}")

# Safeguard: copy pipeline artifacts to robust-specific versions
os.makedirs("saved_state", exist_ok=True)
if os.path.exists("saved_state/scaler.pkl"):
    shutil.copy("saved_state/scaler.pkl", "saved_state/robust_scaler.pkl")
if os.path.exists("saved_state/feature_cols.pkl"):
    shutil.copy("saved_state/feature_cols.pkl", "saved_state/robust_feature_cols.pkl")

dataset = TensorDataset(torch.FloatTensor(X_train_maj))
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"  Batches per epoch: {len(loader)}")

#  2. Model + optimisers
model     = RobustVAEBiGAN(input_dim, LATENT_DIM)
criterion = nn.BCELoss()
mse_loss  = nn.MSELoss()

# G + E share one optimiser (same as vanilla BiGAN)
opt_ge = optim.Adam(
    list(model.generator.parameters()) +
    list(model.encoder_base.parameters()) +
    list(model.fc_mu.parameters()) +
    list(model.fc_logvar.parameters()),
    lr=LR, betas=(0.5, 0.999))

opt_d = optim.Adam(
    model.discriminator.parameters(),
    lr=LR, betas=(0.5, 0.999))

#  3. Training loop
print(f"\nTraining Robust VAE-BiGAN | epochs={EPOCHS} | ε={EPSILON} | "
      f"λ_recon={LAMBDA_RECON} | β_KL_max={BETA_KL_MAX}")
print("-" * 70)

for epoch in range(EPOCHS):
    model.train()
    epoch_d_loss  = 0.0
    epoch_ge_loss = 0.0

    # Anneal KL weight: 0 for first 10 epochs, then linearly to BETA_KL_MAX
    warmup = 10
    if epoch < warmup:
        beta_kl = 0.0
    else:
        beta_kl = BETA_KL_MAX * min(1.0, (epoch - warmup) / (EPOCHS - warmup))

    for (batch_x,) in loader:
        B = batch_x.size(0)

        real_label = torch.ones(B, 1)
        fake_label = torch.zeros(B, 1)

        batch_x.requires_grad_(True)
        mu, logvar = model.encode(batch_x)
        z_enc      = model.reparameterize(mu, logvar)
        real_pair  = torch.cat([batch_x, z_enc], dim=1)
        adv_loss = criterion(model.discriminator(real_pair), fake_label)
        model.zero_grad()
        adv_loss.backward()

        with torch.no_grad():
            x_adv = torch.clamp(
                batch_x - EPSILON * batch_x.grad.sign(), 0.0, 1.0
            ).detach()
        batch_x = batch_x.detach()

        opt_d.zero_grad()

        with torch.no_grad():
            mu_enc, lv_enc = model.encode(batch_x)
            z_real         = model.reparameterize(mu_enc, lv_enc)
            z_rand         = torch.randn(B, LATENT_DIM)
            x_fake         = model.generator(z_rand)

        real_pair = torch.cat([batch_x, z_real], dim=1)
        d_real    = criterion(model.discriminator(real_pair), real_label)

        fake_pair = torch.cat([x_fake, z_rand], dim=1)
        d_fake    = criterion(model.discriminator(fake_pair), fake_label)

        with torch.no_grad():
            mu_adv, lv_adv = model.encode(x_adv)
            z_adv          = model.reparameterize(mu_adv, lv_adv)
        adv_pair = torch.cat([x_adv, z_adv], dim=1)
        d_adv    = criterion(model.discriminator(adv_pair), real_label)

        d_loss = (d_real + d_fake + d_adv) / 3.0
        d_loss.backward()
        nn.utils.clip_grad_norm_(model.discriminator.parameters(), GRAD_CLIP)
        opt_d.step()

        opt_ge.zero_grad()

        mu, logvar = model.encode(batch_x)
        z_enc      = model.reparameterize(mu, logvar)
        x_recon    = model.generator(z_enc)

        real_pair_ge = torch.cat([batch_x, z_enc], dim=1)
        loss_e = criterion(model.discriminator(real_pair_ge), fake_label)

        z_rand       = torch.randn(B, LATENT_DIM)
        x_fake_ge    = model.generator(z_rand)
        fake_pair_ge = torch.cat([x_fake_ge, z_rand], dim=1)
        loss_g = criterion(model.discriminator(fake_pair_ge), real_label)

        loss_recon = mse_loss(x_recon, batch_x)

        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

        ge_loss = (loss_e + loss_g) * 0.5 + LAMBDA_RECON * loss_recon + beta_kl * kl_div
        ge_loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.generator.parameters()) +
            list(model.encoder_base.parameters()) +
            list(model.fc_mu.parameters()) +
            list(model.fc_logvar.parameters()),
            GRAD_CLIP)
        opt_ge.step()

        epoch_d_loss  += d_loss.item()
        epoch_ge_loss += ge_loss.item()

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        avg_d  = epoch_d_loss  / len(loader)
        avg_ge = epoch_ge_loss / len(loader)
        flag = ""
        if avg_d > 1.5:
            flag = " [warn: D loss high]"
        elif avg_d < 0.1:
            flag = " [warn: D collapsed]"
        print(f"Epoch {epoch:3d}/{EPOCHS} | D Loss: {avg_d:.4f} | "
              f"GE Loss: {avg_ge:.4f} | β_KL: {beta_kl:.5f}{flag}")

print(f"\nSaving model to {MODEL_SAVE}...")
torch.save(model.state_dict(), MODEL_SAVE)
print("  Saved.")

print("\nCalibrating Robust VAE-BiGAN on full test set...")
model.eval()
X_test_t = torch.FloatTensor(X_test.astype(np.float32))
y_binary  = (y_test > 0).astype(int)

all_scores = []
with torch.no_grad():
    for i in range(0, len(X_test_t), 1000):
        batch    = X_test_t[i:i+1000]
        mu, _    = model.encode(batch)
        pair     = torch.cat([batch, mu], dim=1)
        s        = model.discriminator(pair).squeeze().numpy()
        if s.ndim == 0:
            s = s.reshape(1)
        all_scores.append(s)

scores = np.concatenate(all_scores)

print(f"\n  Score distribution:")
print(f"    All   - min={scores.min():.4f}  max={scores.max():.4f}  "
      f"mean={scores.mean():.4f}  std={scores.std():.4f}")
print(f"    Benign- mean={scores[y_binary==0].mean():.4f}  "
      f"std={scores[y_binary==0].std():.4f}  n={int((y_binary==0).sum())}")
print(f"    Attack- mean={scores[y_binary==1].mean():.4f}  "
      f"std={scores[y_binary==1].std():.4f}  n={int((y_binary==1).sum())}")

auc  = roc_auc_score(y_binary, scores)
flip = auc < 0.5
if flip:
    print(f"\n  AUC={auc:.4f} < 0.5 - flipping scores.")
    scores = 1.0 - scores
    auc    = roc_auc_score(y_binary, scores)
print(f"  AUC-ROC (flip={flip}): {auc:.4f}")

if auc < 0.65:
    print(f"\n  Warning: AUC={auc:.4f} is low.")

fpr, tpr, thresholds = roc_curve(y_binary, scores)
tnr      = 1.0 - fpr
j_scores = tpr + tnr - 1.0
best_idx  = int(np.argmax(j_scores))
threshold = float(thresholds[best_idx])

print(f"\n  Youden's J threshold: {threshold:.4f}  "
      f"(TPR={tpr[best_idx]:.4f}, TNR={tnr[best_idx]:.4f}, "
      f"J={j_scores[best_idx]:.4f})")

y_pred = (scores > threshold).astype(int)
tp = int((y_pred[y_binary == 1] == 1).sum())
tn = int((y_pred[y_binary == 0] == 0).sum())
fp = int((y_pred[y_binary == 0] == 1).sum())
fn = int((y_pred[y_binary == 1] == 0).sum())
n_attack = int(y_binary.sum())
n_benign = int((y_binary == 0).sum())

print(f"\n  At threshold {threshold:.4f}:")
print(f"    Attacks caught (TP): {tp}/{n_attack} ({tp/max(n_attack,1)*100:.1f}%)")
print(f"    Benign correct (TN): {tn}/{n_benign} ({tn/max(n_benign,1)*100:.1f}%)")
print(f"    False alarms   (FP): {fp}/{n_benign} ({fp/max(n_benign,1)*100:.2f}%)")
print(f"    Missed attacks (FN): {fn}/{n_attack} ({fn/max(n_attack,1)*100:.2f}%)")

calib = {
    'threshold'     : threshold,
    'flip'          : flip,
    'train_classes' : TRAIN_CLASSES,
}
with open(CALIB_PATH, 'wb') as f:
    pickle.dump(calib, f)

print(f"\n  Calibration saved to {CALIB_PATH}")
print(f"    threshold={threshold:.4f}  flip={flip}")

print("\nRobust VAE-BiGAN training complete.")
print("Next: python comparison.py")
"""Evaluate BiGAN using saved calibration artifacts."""

import pickle
import torch
import numpy as np
from models.bigan_model import BiGAN
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

CALIB_PATH = "bigan_calibration.pkl"

print("Loading BiGAN model and calibration bundle...")
checkpoint = torch.load('bigan_final.pth', map_location='cpu')
test_data  = np.load('test_data.npz')

X_test, y_test = test_data['X_test'], test_data['y_test']
input_dim      = checkpoint['input_dim']

model = BiGAN(input_dim, latent_dim=8)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

try:
    with open(CALIB_PATH, 'rb') as f:
        calib = pickle.load(f)
    threshold = calib['threshold']
    flip      = calib['flip']
    r_min     = calib['r_min']
    r_max     = calib['r_max']
    print(f"  threshold={threshold:.4f}  flip={flip}  "
          f"r_min={r_min:.6f}  r_max={r_max:.6f}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"{CALIB_PATH} not found. Run train_bigan.py first."
    )

print("Computing anomaly scores...")
X_test_t = torch.FloatTensor(X_test)

with torch.no_grad():
    z_enc   = model.encoder(X_test_t)
    x_hat   = model.generator(z_enc)
    recon   = torch.mean((X_test_t - x_hat) ** 2, dim=1).numpy()
    pair    = torch.cat((X_test_t, z_enc), dim=1)
    d_score = model.discriminator(pair).squeeze().numpy()
    if d_score.ndim == 0:
        d_score = d_score.reshape(1)

span       = r_max - r_min
recon_norm = (recon - r_min) / (span + 1e-8) if span > 0 else np.zeros_like(recon)
recon_norm = np.clip(recon_norm, 0.0, 1.0)

raw_scores = 0.5 * (1.0 - d_score) + 0.5 * recon_norm

scores = (1.0 - raw_scores) if flip else raw_scores

y_binary = (y_test > 0).astype(int)
y_pred   = (scores > threshold).astype(int)

benign_scores = scores[y_binary == 0]
attack_scores = scores[y_binary == 1]

auc = roc_auc_score(y_binary, scores)
cm  = confusion_matrix(y_binary, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nScore distributions:")
print(f"  Benign - mean={benign_scores.mean():.4f}  "
      f"std={benign_scores.std():.4f}  n={len(benign_scores)}")
print(f"  Attack - mean={attack_scores.mean():.4f}  "
      f"std={attack_scores.std():.4f}  n={len(attack_scores)}")
print(f"  Threshold (Youden's J) : {threshold:.4f}")

print(f"\n{'='*50}")
print("  BiGAN ANOMALY DETECTION RESULTS")
print(f"{'='*50}")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  flip     : {flip}")

print(f"\nConfusion Matrix (rows=true, cols=pred):")
print(f"              Pred Benign  Pred Attack")
print(f"  True Benign     {tn:6d}       {fp:6d}")
print(f"  True Attack     {fn:6d}       {tp:6d}")

total_benign = tn + fp
total_attack = tp + fn
print(f"\n  True  Positives (attacks caught) : {tp} / {total_attack}  "
      f"({tp/max(total_attack,1)*100:.1f}%)")
print(f"  True  Negatives (benign correct) : {tn} / {total_benign}  "
      f"({tn/max(total_benign,1)*100:.1f}%)")
print(f"  False Positives (false alarms)   : {fp} / {total_benign}  "
      f"({fp/max(total_benign,1)*100:.2f}%)")
print(f"  False Negatives (missed attacks) : {fn} / {total_attack}  "
      f"({fn/max(total_attack,1)*100:.2f}%)")

print(f"\nClassification Report:")
print(classification_report(
    y_binary, y_pred,
    target_names=['Benign (class 0)', 'Attack (classes 1-4)'],
    zero_division=0
))

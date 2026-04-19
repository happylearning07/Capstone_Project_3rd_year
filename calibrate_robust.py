"""Calibrate threshold for Robust VAE-BiGAN."""

import torch
import numpy as np
import pickle
from models.robust_model import RobustVAEBiGAN
from sklearn.metrics import roc_auc_score, roc_curve

CALIB_PATH = "robust_calibration.pkl"
MODEL_PATH = "saved_state/robust_vae_bigan_model.pth"

test_data = np.load('test_data.npz')
X_test    = test_data['X_test'].astype(np.float32)
y_test    = test_data['y_test']
input_dim = X_test.shape[1]

model = RobustVAEBiGAN(input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

print("Computing Robust VAE-BiGAN scores on test set...")
X_t = torch.FloatTensor(X_test)

with torch.no_grad():
    all_scores = []
    for i in range(0, len(X_t), 1000):
        batch    = X_t[i:i+1000]
        mu, _    = model.encode(batch)
        s        = model.discriminator(
            torch.cat([batch, mu], dim=1)).squeeze().numpy()
        if s.ndim == 0:
            s = s.reshape(1)
        all_scores.append(s)

scores   = np.concatenate(all_scores)
y_binary = (y_test > 0).astype(int)

print(f"  Scores - min={scores.min():.4f}  max={scores.max():.4f}  "
      f"mean={scores.mean():.4f}  std={scores.std():.4f}")
print(f"  Benign mean={scores[y_binary==0].mean():.4f}  "
      f"Attack mean={scores[y_binary==1].mean():.4f}")

auc  = roc_auc_score(y_binary, scores)
flip = auc < 0.5
if flip:
    print(f"  AUC={auc:.4f} < 0.5 - flipping scores.")
    scores = 1.0 - scores
    auc    = roc_auc_score(y_binary, scores)

print(f"  AUC-ROC (flip={flip}): {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_binary, scores)
tnr      = 1.0 - fpr
j_scores = tpr + tnr - 1.0
best_idx  = int(np.argmax(j_scores))
threshold = float(thresholds[best_idx])

print(f"  Youden's J threshold: {threshold:.4f}  "
      f"(TPR={tpr[best_idx]:.4f}, TNR={tnr[best_idx]:.4f})")

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
    'threshold' : threshold,
    'flip'      : flip,
}
with open(CALIB_PATH, 'wb') as f:
    pickle.dump(calib, f)

print(f"\nCalibration saved to {CALIB_PATH}")
print(f"  threshold={threshold:.4f}  flip={flip}")
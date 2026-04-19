"""Load IDS models and expose a shared predict interface."""

import pickle
import numpy as np
import torch

from models.bigan_model import BiGAN
from models.aae_model   import AAE

BIGAN_PATH  = "bigan_final.pth"
CALIB_PATH  = "bigan_calibration.pkl"
AAE_ENC_PATH = "encoder_final.pth"
AAE_CLF_PATH = "aae_classifier.pkl"


def _align_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad with zeros or truncate X to exactly `target_dim` columns."""
    n = X.shape[1]
    if n == target_dim:
        return X
    if n < target_dim:
        pad = np.zeros((len(X), target_dim - n), dtype=np.float32)
        return np.hstack([X, pad])
    return X[:, :target_dim]


def load_bigan():
    """
    Returns (predict_fn, input_dim) for the BiGAN anomaly detector.

    predict_fn(X) -> (preds, scores)
      preds  : 0=benign, 1=attack  (N,)
      scores : anomaly score [0,1]  (N,)  higher = more anomalous
    """
    ckpt      = torch.load(BIGAN_PATH, map_location="cpu")
    input_dim = ckpt["input_dim"]

    model = BiGAN(input_dim, latent_dim=8)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    try:
        with open(CALIB_PATH, "rb") as f:
            calib = pickle.load(f)
        threshold = calib["threshold"]
        flip      = calib["flip"]
        r_min     = calib["r_min"]
        r_max     = calib["r_max"]
        print(f"BiGAN | calibration loaded: threshold={threshold:.4f}  "
              f"flip={flip}  r_min={r_min:.4f}  r_max={r_max:.4f}")
    except FileNotFoundError:
        threshold, flip, r_min, r_max = 0.5, False, 0.0, 1.0
        print(
            f"Warning: {CALIB_PATH} not found. "
            "Using fallback threshold=0.5 / flip=False. "
            "Re-run train_bigan.py to fix this."
        )

    span = r_max - r_min

    def predict(X: np.ndarray):
        X = np.array(X, dtype=np.float32)
        X = _align_features(X, input_dim)
        x_t = torch.FloatTensor(X)

        with torch.no_grad():
            z_enc   = model.encoder(x_t)
            x_hat   = model.generator(z_enc)
            recon   = torch.mean((x_t - x_hat) ** 2, dim=1).numpy()
            pair    = torch.cat([x_t, z_enc], dim=1)
            d_score = model.discriminator(pair).squeeze().numpy()
            if d_score.ndim == 0:
                d_score = d_score.reshape(1)

        recon_norm = (recon - r_min) / (span + 1e-8) if span > 0 else np.zeros_like(recon)
        recon_norm = np.clip(recon_norm, 0.0, 1.0)

        raw_scores    = 0.5 * (1.0 - d_score) + 0.5 * recon_norm
        anomaly_score = (1.0 - raw_scores) if flip else raw_scores
        preds         = (anomaly_score > threshold).astype(int)

        return preds, anomaly_score

    return predict, input_dim


def load_aae():
    """
    Returns (predict_fn, input_dim) for the AAE + Random Forest classifier.

    predict_fn(X) -> (preds, scores)
      preds  : class label 0-4         (N,)
      scores : RF max class probability (N,)
    """
    ckpt      = torch.load(AAE_ENC_PATH, map_location="cpu")
    input_dim = ckpt["input_dim"]

    aae = AAE(input_dim=input_dim)
    aae.encoder.load_state_dict(ckpt["state_dict"])
    aae.encoder.eval()

    with open(AAE_CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    def predict(X: np.ndarray):
        X = np.array(X, dtype=np.float32)
        X = _align_features(X, input_dim)

        with torch.no_grad():
            Z = aae.encoder(torch.FloatTensor(X)).numpy()

        preds  = clf.predict(Z)
        probas = clf.predict_proba(Z)
        scores = probas.max(axis=1)

        return preds, scores

    return predict, input_dim


MODEL_REGISTRY = {
    "bigan": load_bigan,
    "aae":   load_aae,
}


def get_model(model_type: str):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]()
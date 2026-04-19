"""Utility helpers for gateway model loading and feature alignment."""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("GatewayUtils")

WEIGHTS_DIR            = "gateway/weights"
CGAN_WEIGHTS_PATH      = os.path.join(WEIGHTS_DIR, "cgan_weights.pth")
GAN_SCALER_PATH        = os.path.join(WEIGHTS_DIR, "gan_scaler.pkl")
GAN_LABEL_ENC_PATH     = os.path.join(WEIGHTS_DIR, "gan_label_encoder.pkl")
GAN_ID_TO_CODE_PATH    = os.path.join(WEIGHTS_DIR, "gan_id_to_pipeline_code.pkl")
GAN_FEATURE_COLS_PATH  = os.path.join(WEIGHTS_DIR, "gan_feature_cols.pkl")

PIPELINE_SCALER_PATH        = "saved_state/scaler.pkl"
PIPELINE_FEATURE_COLS_PATH  = "saved_state/feature_cols.pkl"
PIPELINE_REVERSE_MAP_PATH   = "saved_state/reverse_label_map.pkl"


def load_cgan():
    """
    Load trained ConditionalGAN from disk.

    Returns
    -------
    model        : ConditionalGAN (eval mode)
    label_enc    : sklearn LabelEncoder  (class name <-> compact int)
    id_to_code   : dict  compact_int -> pipeline_code
    feature_cols : list  GAN output column names
    scaler       : fitted MinMaxScaler used during GAN training
    """
    from gateway.generator_model import ConditionalGAN

    if not os.path.exists(CGAN_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"cGAN weights not found at {CGAN_WEIGHTS_PATH}.\n"
            "Run: python -m gateway.trainer --data <csv> --model cgan"
        )

    ckpt  = torch.load(CGAN_WEIGHTS_PATH, map_location='cpu')
    model = ConditionalGAN(
        latent_dim  = ckpt['latent_dim'],
        feature_dim = ckpt['feature_dim'],
        num_classes = ckpt['num_classes'],
        embed_dim   = ckpt['embed_dim'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    label_enc, id_to_code, feature_cols, scaler = _load_aux()
    logger.info("cGAN loaded | features=%d | classes=%d",
                ckpt['feature_dim'], ckpt['num_classes'])
    return model, label_enc, id_to_code, feature_cols, scaler


def _load_aux():
    """Load auxiliary pkl files shared by gateway generators."""
    _require(GAN_LABEL_ENC_PATH,    "gan_label_encoder.pkl")
    _require(GAN_ID_TO_CODE_PATH,   "gan_id_to_pipeline_code.pkl")
    _require(GAN_FEATURE_COLS_PATH, "gan_feature_cols.pkl")
    _require(GAN_SCALER_PATH,       "gan_scaler.pkl")

    with open(GAN_LABEL_ENC_PATH,    'rb') as f: le           = pickle.load(f)
    with open(GAN_ID_TO_CODE_PATH,   'rb') as f: id_to_code   = pickle.load(f)
    with open(GAN_FEATURE_COLS_PATH, 'rb') as f: feature_cols = pickle.load(f)
    with open(GAN_SCALER_PATH,       'rb') as f: scaler       = pickle.load(f)
    return le, id_to_code, feature_cols, scaler


def _require(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Train the gateway generator first with gateway/trainer.py"
        )


def name_to_compact_id(label_name: str, label_enc) -> int:
    """
    Translate a human-readable class name ('Okiru') to GAN-internal compact int.
    Raises ValueError with a helpful message if not found.
    """
    classes = list(label_enc.classes_)
    if label_name not in classes:
        lower_map = {c.lower(): c for c in classes}
        matched   = lower_map.get(label_name.lower())
        if matched:
            logger.warning("Label '%s' not found - using '%s'", label_name, matched)
            label_name = matched
        else:
            raise ValueError(
                f"Unknown label '{label_name}'.\nAvailable classes: {classes}"
            )
    return int(label_enc.transform([label_name])[0])


def compact_id_to_pipeline_code(compact_id: int, id_to_code: dict) -> int:
    """Translate GAN compact id to IDS pipeline numeric code."""
    if compact_id not in id_to_code:
        raise KeyError(f"compact_id {compact_id} not in id_to_code mapping.")
    return id_to_code[compact_id]


def all_class_names(label_enc) -> list:
    """Return sorted list of all class names known to the trained GAN."""
    return sorted(label_enc.classes_)


def align_to_pipeline(X_gan: np.ndarray,
                      gan_cols: list,
                      pipeline_cols: list = None) -> pd.DataFrame:
    """
    Align GAN output columns to the IDS pipeline column order.

    Uses fast path when columns already match; otherwise reorders and fills.

    Parameters
    ----------
    X_gan        : (N, len(gan_cols)) array of generated samples
    gan_cols     : column names in GAN output order
    pipeline_cols: expected column names for the IDS models.
                   If None, tries to load from saved_state/feature_cols.pkl.

    Returns
    -------
    X_aligned    : DataFrame (N, len(pipeline_cols)) - ready for IDS model input
    """
    X_df = pd.DataFrame(X_gan, columns=gan_cols)

    if pipeline_cols is None:
        if not os.path.exists(PIPELINE_FEATURE_COLS_PATH):
            logger.warning(
                "saved_state/feature_cols.pkl not found - "
                "returning GAN output unchanged (columns may not match IDS model)."
            )
            return X_df
        with open(PIPELINE_FEATURE_COLS_PATH, 'rb') as f:
            pipeline_cols = pickle.load(f)

    if gan_cols == pipeline_cols:
        return X_df[pipeline_cols]

    logger.warning(
        "GAN cols (%d) != pipeline cols (%d) - re-aligning. "
        "Retrain the GAN to fix this permanently.",
        len(gan_cols), len(pipeline_cols)
    )
    matched = len([c for c in pipeline_cols if c in X_df.columns])
    filled = len(pipeline_cols) - matched
    result = X_df.reindex(columns=pipeline_cols, fill_value=0.0).astype(np.float32)

    logger.warning("Column alignment: %d matched, %d filled with zeros.",
                   matched, filled)
    return result


def sample_noise(n: int, latent_dim: int,
                 device: str = 'cpu') -> torch.Tensor:
    """Sample standard Gaussian noise z ~ N(0, I)."""
    return torch.randn(n, latent_dim, device=device)


def partial_latent_noise(n: int, latent_dim: int,
                          noise_fraction: float = 0.5,
                          device: str = 'cpu') -> torch.Tensor:
    """
    Early-stage simulation via partial latent trajectory.
    Only `noise_fraction` of latent dims are active; the rest are zeroed.
    Pushes generation toward class mean for early-stage simulation.
    """
    z    = torch.randn(n, latent_dim, device=device)
    mask = torch.zeros_like(z)
    k    = max(1, int(latent_dim * noise_fraction))
    indices = torch.randperm(latent_dim)[:k]
    mask[:, indices] = 1.0
    return z * mask


def inject_feature_noise(X: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """
    Add Gaussian noise to scaled features.
    Clips to [0, 1] to stay within MinMaxScaler range.
    """
    noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
    return np.clip(X + noise, 0.0, 1.0)


def truncate_sequence(X: np.ndarray, keep_fraction: float = 0.5) -> np.ndarray:
    """
    Zero out the latter (1 - keep_fraction) of the feature vector.
    Mimics partial flow capture.
    """
    n_keep   = max(1, int(X.shape[1] * keep_fraction))
    X_trunc  = X.copy()
    X_trunc[:, n_keep:] = 0.0
    return X_trunc


def load_pipeline_reverse_map() -> dict:
    """
    Load saved_state/reverse_label_map.pkl: {name -> pipeline int code}.

    This file is written by utils/preprocessing.py (load_and_clean).
    """
    if os.path.exists(PIPELINE_REVERSE_MAP_PATH):
        with open(PIPELINE_REVERSE_MAP_PATH, 'rb') as f:
            rmap = pickle.load(f)
        logger.info("reverse_label_map loaded: %s", rmap)
        return rmap

    logger.error(
        "saved_state/reverse_label_map.pkl NOT FOUND.\n"
        "This file is required for correct label translation.\n"
        "Fix: run a preprocessing bootstrap (load_and_clean) to create the "
        "shared pipeline artifacts."
    )
    return {}
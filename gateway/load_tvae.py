"""
gateway/load_tvae.py
--------------------
Utility to load a trained TVAE from disk.
Mirrors load_cgan() in gateway/utils.py exactly so Gateway can use either.

Usage
-----
    from gateway.load_tvae import load_tvae
    model, label_enc, id_to_code, feature_cols, scaler = load_tvae()
"""

import os
import pickle
import logging
import torch

logger = logging.getLogger("LoadTVAE")

WEIGHTS_DIR            = "gateway/weights"
TVAE_WEIGHTS_PATH      = os.path.join(WEIGHTS_DIR, "tvae_weights.pth")
TVAE_LABEL_ENC_PATH    = os.path.join(WEIGHTS_DIR, "tvae_label_encoder.pkl")
TVAE_ID_TO_CODE_PATH   = os.path.join(WEIGHTS_DIR, "tvae_id_to_pipeline_code.pkl")
TVAE_FEATURE_COLS_PATH = os.path.join(WEIGHTS_DIR, "tvae_feature_cols.pkl")
TVAE_SCALER_PATH       = os.path.join(WEIGHTS_DIR, "tvae_scaler.pkl")


def load_tvae():
    """
    Load trained ConditionalTVAE from disk.

    Returns
    -------
    model        : ConditionalTVAE (eval mode)
    label_enc    : sklearn LabelEncoder  (class name ↔ compact int)
    id_to_code   : dict  compact_int -> pipeline_code
    feature_cols : list  TVAE output column names (same as pipeline)
    scaler       : fitted MinMaxScaler
    """
    from gateway.tvae_model import ConditionalTVAE

    if not os.path.exists(TVAE_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"TVAE weights not found at {TVAE_WEIGHTS_PATH}.\n"
            "Run: python -m gateway.train_tvae --data <csv>"
        )

    ckpt = torch.load(TVAE_WEIGHTS_PATH, map_location="cpu")
    model = ConditionalTVAE(
        feature_dim  = ckpt["feature_dim"],
        num_classes  = ckpt["num_classes"],
        latent_dim   = ckpt["latent_dim"],
        embed_dim    = ckpt["embed_dim"],
        hidden_dims  = tuple(ckpt["hidden_dims"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    for path in [TVAE_LABEL_ENC_PATH, TVAE_ID_TO_CODE_PATH,
                 TVAE_FEATURE_COLS_PATH, TVAE_SCALER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required TVAE auxiliary file not found: {path}\n"
                "Run: python -m gateway.train_tvae --data <csv>"
            )

    with open(TVAE_LABEL_ENC_PATH,    "rb") as f: label_enc    = pickle.load(f)
    with open(TVAE_ID_TO_CODE_PATH,   "rb") as f: id_to_code   = pickle.load(f)
    with open(TVAE_FEATURE_COLS_PATH, "rb") as f: feature_cols = pickle.load(f)
    with open(TVAE_SCALER_PATH,       "rb") as f: scaler       = pickle.load(f)

    logger.info("TVAE loaded | features=%d | classes=%d | latent=%d",
                ckpt["feature_dim"], ckpt["num_classes"], ckpt["latent_dim"])
    return model, label_enc, id_to_code, feature_cols, scaler
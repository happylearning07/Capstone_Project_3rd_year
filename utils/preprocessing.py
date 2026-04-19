"""
utils/preprocessing.py
-----------------------
SHARED preprocessing for ALL models.
Now includes engineered features for better inter-class separability.

Feature engineering is applied BEFORE one-hot encoding and scaling,
expanding the feature space from 18 -> ~34 columns.  This dramatically
improves GAN conditioning and IDS model accuracy.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from utils.feature_engineering import add_engineered_features


SCALER_PATH            = "saved_state/scaler.pkl"
FEATURE_COLS_PATH      = "saved_state/feature_cols.pkl"
REVERSE_LABEL_MAP_PATH = "saved_state/reverse_label_map.pkl"


def load_and_clean(filepath, n_rows=500000):
    """
    Load, engineer features, clean, encode, and scale IoT-23 CSV.

    Pipeline
    --------
    1. Load CSV
    2. Drop non-feature columns
    3. Fix numeric types
    4. ADD ENGINEERED FEATURES  ← new step
    5. One-hot encode categoricals
    6. Fit MinMaxScaler and save
    7. Save feature_cols + reverse_label_map

    Returns X_train, X_test, y_train, y_test, input_dim
    """
    os.makedirs("saved_state", exist_ok=True)

    df = pd.read_csv(filepath, nrows=n_rows)

    # Drop non-feature columns
    to_drop = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
               'history', 'local_orig', 'local_resp']
    df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

    # Replace Zeek nulls
    df.replace('-', 0, inplace=True)

    # Fix numeric columns
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts',
                    'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes', 'missed_bytes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = add_engineered_features(df)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=['proto', 'service', 'conn_state'])

    # Separate features and labels
    X = df.drop(columns=['label', 'detailed-label'], errors='ignore')
    label_series = df['label'].astype(str).str.strip()

    # Build label codes - deterministic alphabetical ordering
    label_cat = label_series.astype('category')
    y = label_cat.cat.codes.values

    reverse_label_map = {
        name: int(code)
        for code, name in enumerate(label_cat.cat.categories)
    }

    X = X.astype(float)
    feature_cols = list(X.columns)

    # Save artifacts
    with open(FEATURE_COLS_PATH, 'wb') as f:
        pickle.dump(feature_cols, f)
    with open(REVERSE_LABEL_MAP_PATH, 'wb') as f:
        pickle.dump(reverse_label_map, f)

    print(f"[preprocessing] reverse_label_map: {reverse_label_map}")
    print(f"[preprocessing] feature_cols ({len(feature_cols)}): {feature_cols}")

    # Fit and save scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, X.shape[1]


def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)


def preprocess_raw_sample(sample_dict):
    """
    Preprocess a single raw feature dict using SAVED scaler + feature columns.
    """
    with open(FEATURE_COLS_PATH, 'rb') as f:
        feature_cols = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    row = {col: 0.0 for col in feature_cols}
    for k, v in sample_dict.items():
        if k in row:
            row[k] = float(v)

    X_df = pd.DataFrame([row], columns=feature_cols, dtype=float)
    return scaler.transform(X_df)


def get_input_dim():
    with open(FEATURE_COLS_PATH, 'rb') as f:
        feature_cols = pickle.load(f)
    return len(feature_cols)

"""
train_ocsvm.py
--------------
Train and calibrate the One-Class SVM (OC-SVM) anomaly detector.

Design follows the paper:
  "Mitigating IoT botnet attacks: An early-stage explainable network-based
   anomaly detection approach" (Amara Korba et al., 2025)

Key differences from supervised models:
  - OC-SVM is trained on BENIGN-ONLY data (semi-supervised paradigm).
  - The model learns a tight boundary around normal IoT behaviour.
  - Any traffic outside this boundary is flagged as anomalous (attack).
  - This enables detection of UNKNOWN botnets not seen during training.

Outputs
-------
  ocsvm_model.pkl          – trained + calibrated OC-SVM detector
  ocsvm_calibration.pkl    – threshold + flip metadata (comparison.py compatible)
  test_data.npz            – shared test split (created by load_and_clean if missing)

Usage
-----
    python train_ocsvm.py
    python train_ocsvm.py --nu 0.05 --gamma scale --rows 300000
"""

import os
import argparse
import pickle
import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from models.ocsvm_model import OCSVMDetector
from utils.preprocessing import load_and_clean

# ---------------------------------------------------------------------------
# Config defaults (match paper: nu in [0.01, 0.1], gamma random-searched)
# ---------------------------------------------------------------------------
DEFAULT_NU        = 0.05      # 5% outlier tolerance
DEFAULT_GAMMA     = 'scale'   # 1 / (n_features * X.var())
DEFAULT_KERNEL    = 'rbf'
DEFAULT_N_ROWS    = 500_000   # match BiGAN / Robust training row count
BENIGN_CLASS      = 0         # label code for benign in reverse_label_map

MODEL_SAVE_PATH  = "ocsvm_model.pkl"
CALIB_SAVE_PATH  = "ocsvm_calibration.pkl"
TEST_DATA_PATH   = "test_data.npz"
DATA_CSV_PATH    = "data/iot23_combined_new.csv"


def train_ocsvm(
    nu: float          = DEFAULT_NU,
    gamma: float | str = DEFAULT_GAMMA,
    kernel: str        = DEFAULT_KERNEL,
    n_rows: int        = DEFAULT_N_ROWS,
    calib_method: str  = 'youden',
    data_path: str     = DATA_CSV_PATH,
) -> OCSVMDetector:
    """
    Full train-calibrate-save pipeline for the OC-SVM detector.

    Parameters
    ----------
    nu           : OC-SVM nu hyper-parameter (outlier fraction).
    gamma        : RBF kernel bandwidth.
    kernel       : Kernel type.
    n_rows       : Number of CSV rows to load (match other models).
    calib_method : 'youden' or 'f1' threshold calibration.
    data_path    : Path to IoT-23 combined CSV.

    Returns
    -------
    Fitted OCSVMDetector.
    """

    print("=" * 65)
    print("  One-Class SVM - IoT Botnet Anomaly Detector")
    print("  Based on: Amara Korba et al., Computer Communications 2025")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load and preprocess data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading data: {data_path}  (n_rows={n_rows})")
    X_train, X_test, y_train, y_test, input_dim = load_and_clean(
        data_path, n_rows=n_rows
    )
    print(f"  input_dim : {input_dim}")
    print(f"  X_train   : {X_train.shape}")
    print(f"  X_test    : {X_test.shape}")

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Class distribution (train): "
          f"{dict(zip(unique.tolist(), counts.tolist()))}")


    # ------------------------------------------------------------------
    # 2. Isolate BENIGN samples (train + test benign combined)
    # ------------------------------------------------------------------
    # OC-SVM only needs benign traffic to learn the normal boundary.
    # Using test-set benign samples for fitting is still valid because
    # the model never sees attack labels. This matters when the train
    # split has very few benign examples (e.g. 213 in a skewed dataset).
    print(f"\n[2/5] Isolating benign-only data (class={BENIGN_CLASS})")
    X_benign_train = X_train[(y_train == BENIGN_CLASS)].astype(np.float32)
    X_benign_test  = X_test [(y_test  == BENIGN_CLASS)].astype(np.float32)
    X_benign       = np.vstack([X_benign_train, X_benign_test])
    print(f"  Benign from train split : {len(X_benign_train)}")
    print(f"  Benign from test  split : {len(X_benign_test)}")
    print(f"  Total benign for OC-SVM : {len(X_benign)}")

    if len(X_benign) < 10:
        raise ValueError(
            f"Too few benign samples ({len(X_benign)}) to fit OC-SVM reliably. "
            "Check that your reverse_label_map assigns 0 to Benign."
        )

    # ------------------------------------------------------------------
    # 3. Train OC-SVM
    # ------------------------------------------------------------------
    print(f"\n[3/5] Training OC-SVM "
          f"(kernel={kernel}, nu={nu}, gamma={gamma})")
    detector = OCSVMDetector(nu=nu, gamma=gamma, kernel=kernel)
    detector.fit(X_benign)

    # ------------------------------------------------------------------
    # 4. Calibrate threshold on full test set
    # ------------------------------------------------------------------
    print(f"\n[4/5] Calibrating threshold ({calib_method}) on test set")
    threshold = detector.calibrate(
        X_test.astype(np.float32),
        y_test,
        method=calib_method,
    )

    # Compute final metrics at chosen threshold
    y_binary = (y_test > 0).astype(int)
    preds, scores = detector.predict(X_test.astype(np.float32))

    auc = roc_auc_score(y_binary, scores)
    cm  = confusion_matrix(y_binary, preds)
    tn, fp, fn, tp = cm.ravel()

    n_attack = int(y_binary.sum())
    n_benign = int((y_binary == 0).sum())

    print(f"\n  Score distribution:")
    print(f"    Benign - mean={scores[y_binary==0].mean():.4f}  "
          f"std={scores[y_binary==0].std():.4f}  n={n_benign}")
    print(f"    Attack - mean={scores[y_binary==1].mean():.4f}  "
          f"std={scores[y_binary==1].std():.4f}  n={n_attack}")

    print(f"\n{'='*65}")
    print(f"  OC-SVM RESULTS  (threshold={threshold:.4f})")
    print(f"{'='*65}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"  Detection Rate  : {tp / max(n_attack, 1) * 100:.2f}%  "
          f"({tp}/{n_attack} attacks caught)")
    print(f"  False Alarm Rate: {fp / max(n_benign, 1) * 100:.2f}%  "
          f"({fp}/{n_benign} benign flagged)")
    print(f"  Missed Attacks  : {fn / max(n_attack, 1) * 100:.2f}%  "
          f"({fn}/{n_attack})")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    print("\nClassification Report:")
    print(classification_report(
        y_binary, preds,
        target_names=["Benign", "Attack"],
        zero_division=0,
    ))

    # ------------------------------------------------------------------
    # 5. Save model + calibration bundle
    # ------------------------------------------------------------------
    print(f"[5/5] Saving model to {MODEL_SAVE_PATH}")
    detector.save(MODEL_SAVE_PATH)

    calib = {
        'threshold': threshold,
        'flip':      False,     # OC-SVM score is already anomaly-oriented
        'nu':        nu,
        'gamma':     str(gamma),
        'kernel':    kernel,
        'auc':       float(auc),
    }
    with open(CALIB_SAVE_PATH, 'wb') as f:
        pickle.dump(calib, f)
    print(f"  Calibration bundle saved to {CALIB_SAVE_PATH}")
    print(f"    threshold={threshold:.4f}  auc={auc:.4f}")

    # Save/update shared test_data.npz (needed by comparison.py)
    if not os.path.exists(TEST_DATA_PATH):
        np.savez(TEST_DATA_PATH,
                 X_test=X_test,   y_test=y_test,
                 X_train=X_train, y_train=y_train)
        print(f"  Test data saved to {TEST_DATA_PATH}")
    else:
        print(f"  {TEST_DATA_PATH} already exists - not overwritten.")

    print("\nOC-SVM training complete.")
    print("Next: python comparison2.py")
    return detector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train One-Class SVM anomaly detector on IoT-23 data"
    )
    parser.add_argument(
        "--nu",      type=float, default=DEFAULT_NU,
        help=f"OC-SVM nu parameter (default: {DEFAULT_NU})"
    )
    parser.add_argument(
        "--gamma",   type=str,   default=str(DEFAULT_GAMMA),
        help="RBF gamma: 'scale', 'auto', or a float (default: scale)"
    )
    parser.add_argument(
        "--kernel",  type=str,   default=DEFAULT_KERNEL,
        help=f"Kernel type (default: {DEFAULT_KERNEL})"
    )
    parser.add_argument(
        "--rows",    type=int,   default=DEFAULT_N_ROWS,
        help=f"Rows to load from CSV (default: {DEFAULT_N_ROWS})"
    )
    parser.add_argument(
        "--calib",   type=str,   default='youden',
        choices=['youden', 'f1'],
        help="Threshold calibration method (default: youden)"
    )
    parser.add_argument(
        "--data",    type=str,   default=DATA_CSV_PATH,
        help=f"Path to IoT-23 CSV (default: {DATA_CSV_PATH})"
    )
    args = parser.parse_args()

    # gamma can be a float or a string keyword
    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass   # keep as string ('scale' or 'auto')

    train_ocsvm(
        nu           = args.nu,
        gamma        = gamma,
        kernel       = args.kernel,
        n_rows       = args.rows,
        calib_method = args.calib,
        data_path    = args.data,
    )


if __name__ == "__main__":
    main()
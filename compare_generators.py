"""Compare GMM, TVAE, and cGAN generators."""

import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from run_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO,
                    format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("CompareGenerators")

W = 110


def _section(title):
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)


def _divider():
    print("  " + "-" * (W - 4))


def _load_real_data():
    data = np.load("test_data.npz")
    X    = data["X_test"].astype(np.float32)
    y    = data["y_test"]
    try:
        with open("saved_state/reverse_label_map.pkl", "rb") as f:
            rev_map = pickle.load(f)
        class_names = {v: k for k, v in rev_map.items()}
    except FileNotFoundError:
        class_names = {i: str(i) for i in range(5)}
    return X, y, class_names


def _load_gmm():
    from gateway.gmm_generator import GMMGenerator
    return GMMGenerator.load()


def _load_tvae():
    from gateway.load_tvae import load_tvae
    return load_tvae()


def _gmm_generate_class(gmm, cls_name, n):
    return gmm.sample_class(cls_name, n)


def _tvae_generate_class(model, le, cls_name, n):
    from gateway.utils import name_to_compact_id
    cid    = name_to_compact_id(cls_name, le)
    labels = torch.full((n,), cid, dtype=torch.long)
    with torch.no_grad():
        z   = torch.randn(n, model.latent_dim)
        out = model.decode(z, labels).numpy()
    return np.clip(out, 0.0, 1.0)


def compute_quality_metrics(X_gen, X_real, feat_names=None):
    """Compute all quality metrics. Returns dict."""
    if len(X_real) < 5 or len(X_gen) < 5:
        return {"mae": float("nan"), "pearson_r": float("nan"),
                "std_ratio": float("nan"), "mean_kl": float("nan")}

    gen_mean  = X_gen.mean(axis=0)
    real_mean = X_real.mean(axis=0)
    gen_std   = X_gen.std(axis=0)
    real_std  = X_real.std(axis=0)

    mae = float(np.abs(gen_mean - real_mean).mean())

    if real_mean.std() < 1e-9 or gen_mean.std() < 1e-9:
        pearson_r = 0.0
    else:
        r = np.corrcoef(real_mean, gen_mean)[0, 1]
        pearson_r = float(r) if not np.isnan(r) else 0.0

    safe_real_std = np.where(real_std > 1e-9, real_std, 1e-9)
    std_ratio     = float((gen_std / safe_real_std).mean())

    kl_divs = []
    for f in range(min(X_gen.shape[1], 39)):
        bins = np.linspace(0, 1, 21)
        p, _ = np.histogram(X_real[:, f], bins=bins, density=True)
        q, _ = np.histogram(X_gen[:, f],  bins=bins, density=True)
        p    = p + 1e-9; q = q + 1e-9
        p    = p / p.sum(); q = q / q.sum()
        kl_divs.append(float(np.sum(p * np.log(p / q))))
    mean_kl = float(np.mean(kl_divs))

    # Top-5 worst features
    feat_mae = np.abs(gen_mean - real_mean)
    top5_idx = np.argsort(feat_mae)[-5:][::-1]
    if feat_names:
        top5 = [(feat_names[i], float(feat_mae[i])) for i in top5_idx
                if i < len(feat_names)]
    else:
        top5 = [(f"f{i}", float(feat_mae[i])) for i in top5_idx]

    return {
        "mae":       mae,
        "pearson_r": pearson_r,
        "std_ratio": std_ratio,
        "mean_kl":   mean_kl,
        "top5_worst": top5,
    }


def _load_bigan():
    from models.bigan_model import BiGAN
    ckpt  = torch.load("bigan_final.pth", map_location="cpu")
    model = BiGAN(ckpt["input_dim"], latent_dim=8)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    with open("bigan_calibration.pkl", "rb") as f:
        calib = pickle.load(f)
    return model, calib


def _predict_bigan(model, calib, X):
    X_t = torch.FloatTensor(X.astype(np.float32))
    with torch.no_grad():
        z     = model.encoder(X_t)
        recon = torch.mean((X_t - model.generator(z)) ** 2, dim=1).numpy()
        d     = model.discriminator(torch.cat([X_t, z], dim=1)).squeeze().numpy()
        if d.ndim == 0: d = d.reshape(1)
    span  = calib["r_max"] - calib["r_min"]
    rn    = np.clip((recon - calib["r_min"]) / (span + 1e-8), 0, 1)
    raw   = 0.5 * (1 - d) + 0.5 * rn
    scores = (1.0 - raw) if calib["flip"] else raw
    return (scores > calib["threshold"]).astype(int), scores


def _blend_values(step=0.1):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    return [float(f"{v:.1f}") for v in vals]


def _compare_generator_matrix(attack_classes=None, n_eval=120):
    """
    Run detection matrix over:
      - generators: none (pure GMM), tvae, cgan
      - blends: 0.0, 0.1, ..., 1.0
      - modes: stream, attack, adversarial
    Uses BiGAN IDS model for consistent comparison.
    """
    _section("[2] DETECTION MATRIX - generators x blends x modes")
    print("""
  Running BiGAN detection-rate matrix for:
    generators = [none, tvae, cgan]
    gan_blend  = [0.0, 0.1, ..., 1.0]
    modes      = [stream, attack, adversarial]
    """)

    generators = ["none", "tvae", "cgan"]
    modes = ["stream", "attack", "adversarial"]
    blends = _blend_values(0.1)
    attack_name = attack_classes[0] if attack_classes else "DDoS"

    col = 10
    header = (f"  {'Generator':<10} {'Mode':<12} {'Blend':>7} "
              f"{'DR%':>{col}} {'FAR%':>{col}} {'Acc%':>{col}}")
    print(header)
    _divider()

    rows = []

    for gen in generators:
        for mode in modes:
            for blend in blends:
                # For pure GMM baseline, blend alpha has no effect.
                effective_blend = blend if gen != "none" else 0.0
                try:
                    results, _, _, _ = run_pipeline(
                        mode=mode,
                        model_type="bigan",
                        generator_type=gen,
                        n_samples=n_eval,
                        attack_type=attack_name if mode in ("attack", "adversarial") else None,
                        stage="full",
                        verbose=False,
                        export_csv=None,
                        epsilon=None,
                        benign_ratio=0.70,
                        gan_blend_alpha=effective_blend,
                    )

                    total = len(results)
                    n_attacks = sum(1 for r in results if r.get("true_label", 0) > 0)
                    n_benign = total - n_attacks
                    attacks_detected = sum(
                        1 for r in results
                        if r.get("true_label", 0) > 0 and r.get("detected", False)
                    )
                    false_alarms = sum(
                        1 for r in results
                        if r.get("true_label", 0) == 0 and r.get("detected", False)
                    )
                    dr = (attacks_detected / n_attacks * 100.0) if n_attacks else 0.0
                    far = (false_alarms / n_benign * 100.0) if n_benign else 0.0
                    acc = ((attacks_detected + n_benign - false_alarms) / total * 100.0) if total else 0.0

                    print(f"  {gen:<10} {mode:<12} {blend:>7.1f} "
                          f"{dr:>{col}.1f} {far:>{col}.1f} {acc:>{col}.1f}")
                    rows.append({
                        "generator": gen,
                        "mode": mode,
                        "blend_requested": blend,
                        "blend_effective": effective_blend,
                        "attack_class": attack_name if mode in ("attack", "adversarial") else "",
                        "n_eval": n_eval,
                        "detection_rate_pct": dr,
                        "false_alarm_rate_pct": far,
                        "accuracy_pct": acc,
                        "status": "ok",
                    })
                except Exception as e:
                    print(f"  {gen:<10} {mode:<12} {blend:>7.1f} "
                          f"{'ERR':>{col}} {'ERR':>{col}} {'ERR':>{col}}")
                    logger.warning("Matrix run failed | gen=%s mode=%s blend=%.1f: %s",
                                   gen, mode, blend, e)
                    rows.append({
                        "generator": gen,
                        "mode": mode,
                        "blend_requested": blend,
                        "blend_effective": effective_blend,
                        "attack_class": attack_name if mode in ("attack", "adversarial") else "",
                        "n_eval": n_eval,
                        "detection_rate_pct": np.nan,
                        "false_alarm_rate_pct": np.nan,
                        "accuracy_pct": np.nan,
                        "status": f"error: {e}",
                    })

    return rows


def run_comparison(attack_classes=None, n_gen=500, run_detection=True,
                   run_matrix=True, csv_path="generator_comparison_results.csv"):

    print(f"\n{'='*W}")
    print(f"{'GMM vs TVAE vs cGAN - Traffic Generator Comparison':^{W}}")
    print(f"{'='*W}")

    print("  Comparing quality metrics and detection behavior across generators.")

    # Load data
    X_real, y_real, class_names = _load_real_data()
    try:
        with open("saved_state/feature_cols.pkl", "rb") as f:
            feat_names = pickle.load(f)
    except Exception:
        feat_names = None

    print(f"  Real test data: {len(X_real)} samples | {X_real.shape[1]} features")

    # Load GMM
    print("\n  Loading generators...")
    gmm = _load_gmm()
    gmm_names = list(gmm.gmms.keys())

    # Load TVAE
    tvae_ok = False
    tvae_model = tvae_le = None
    try:
        tvae_model, tvae_le, _, _, _ = _load_tvae()
        tvae_ok = True
        print("  TVAE loaded")
    except FileNotFoundError:
        print("  TVAE not found - run: python -m gateway.train_tvae --data <csv>")

    # Generate
    gen_gmm  = {}
    gen_tvae = {}
    for cls_name in gmm_names:
        gen_gmm[cls_name] = _gmm_generate_class(gmm, cls_name, n_gen)
        if tvae_ok:
            try:
                gen_tvae[cls_name] = _tvae_generate_class(
                    tvae_model, tvae_le, cls_name, n_gen)
            except Exception as e:
                logger.warning("TVAE failed for %s: %s", cls_name, e)

    if attack_classes is None:
        attack_classes = [n for n in class_names.values() if n != "Benign"]

    _section("[1] SAMPLE QUALITY - Generated vs Real Feature Distributions")

    print(f"\n  {'Metric':<16} Description")
    print(f"  {'MAE':<16} Mean |gen_feature_mean - real_feature_mean| (lower=better)")
    print(f"  {'Pearson r':<16} Correlation of feature means gen vs real (higher=better, 1.0=perfect)")
    print(f"  {'Std Ratio':<16} std(gen)/std(real) - 1.0=right diversity, <0.3=too uniform")
    print(f"  {'KL Div':<16} Mean KL divergence of marginal distributions (lower=better)")

    qual_gmm  = []
    qual_tvae = []

    col = 12
    for code, cls_name in sorted(class_names.items()):
        mask   = y_real == code
        X_cls  = X_real[mask]
        n_real = len(X_cls)
        if n_real < 5:
            continue

        print(f"\n  -- {cls_name}  (n_real={n_real}) " + "-" * 50)
        print(f"  {'Generator':<12} {'MAE':>{col}} {'Pearson r':>{col}} "
              f"{'Std Ratio':>{col}} {'KL Div':>{col}}")
        _divider()

        if cls_name in gen_gmm:
            m = compute_quality_metrics(gen_gmm[cls_name], X_cls, feat_names)
            qual_gmm.append(m)
            print(f"  {'GMM':<12} "
                  f"{m['mae']:>{col}.5f} "
                  f"{m['pearson_r']:>{col}.4f} "
                  f"{m['std_ratio']:>{col}.4f} "
                  f"{m['mean_kl']:>{col}.4f}")
            if m.get("top5_worst"):
                print(f"  GMM worst features: "
                      + ", ".join(f"{n}({v:.4f})" for n, v in m["top5_worst"]))

        if tvae_ok and cls_name in gen_tvae:
            m = compute_quality_metrics(gen_tvae[cls_name], X_cls, feat_names)
            qual_tvae.append(m)
            print(f"  {'TVAE':<12} "
                  f"{m['mae']:>{col}.5f} "
                  f"{m['pearson_r']:>{col}.4f} "
                  f"{m['std_ratio']:>{col}.4f} "
                  f"{m['mean_kl']:>{col}.4f}")
            if m.get("top5_worst"):
                print(f"  TVAE worst features: "
                      + ", ".join(f"{n}({v:.4f})" for n, v in m["top5_worst"]))

    # Summary
    _section("[1] QUALITY SUMMARY")
    print(f"  {'Generator':<12} {'Avg MAE':>10} {'Avg Pearson r':>14} "
          f"{'Avg Std Ratio':>14} {'Avg KL':>10}")
    _divider()

    def _summarise(label, ms):
        if not ms:
            return
        avg_mae = np.nanmean([m["mae"] for m in ms])
        avg_r   = np.nanmean([m["pearson_r"] for m in ms])
        avg_sr  = np.nanmean([m["std_ratio"] for m in ms])
        avg_kl  = np.nanmean([m["mean_kl"] for m in ms])
        print(f"  {label:<12} {avg_mae:>10.5f} {avg_r:>14.4f} "
              f"{avg_sr:>14.4f} {avg_kl:>10.4f}")
        return avg_mae, avg_r, avg_sr, avg_kl

    r_gmm  = _summarise("GMM",  qual_gmm)
    r_tvae = _summarise("TVAE", qual_tvae) if qual_tvae else None

    if r_gmm and r_tvae:
        print()
        winners = []
        if r_gmm[0] < r_tvae[0]: winners.append("GMM (lower MAE)")
        else: winners.append("TVAE (lower MAE)")
        if r_gmm[1] > r_tvae[1]: winners.append("GMM (higher Pearson r)")
        else: winners.append("TVAE (higher Pearson r)")
        if abs(r_gmm[2] - 1.0) < abs(r_tvae[2] - 1.0): winners.append("GMM (better std ratio)")
        else: winners.append("TVAE (better std ratio)")
        if r_gmm[3] < r_tvae[3]: winners.append("GMM (lower KL)")
        else: winners.append("TVAE (lower KL)")
        print(f"  Per-metric winners: {', '.join(winners)}")

    if not run_detection:
        _print_summary()
        return

    matrix_rows = []
    if run_matrix:
        matrix_rows = _compare_generator_matrix(
            attack_classes=attack_classes,
            n_eval=min(120, n_gen)
        )
    else:
        _section("[2] DETECTION MATRIX")
        print("  Skipped (--skip-matrix).")

    if matrix_rows:
        df = pd.DataFrame(matrix_rows)
        df.to_csv(csv_path, index=False)
        print(f"\n  Matrix results saved to: {csv_path}")

    _print_summary()


def _print_summary():
    _section("SUMMARY")
    print("""
  - Use quality metrics (MAE, Pearson r, std ratio, KL) to compare generators.
  - Use the detection matrix to assess deployment impact across mode/blend settings.
  - Keep a GMM baseline for stable benchmarking.
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare GMM vs TVAE vs cGAN traffic generators")
    parser.add_argument("--attack",       type=str, default=None)
    parser.add_argument("--n",            type=int, default=500)
    parser.add_argument("--no-detection", action="store_true")
    parser.add_argument("--skip-matrix",  action="store_true")
    parser.add_argument("--csv",          type=str,
                        default="generator_comparison_results.csv",
                        help="Output CSV path for matrix results.")
    args = parser.parse_args()

    run_comparison(
        attack_classes = [args.attack] if args.attack else None,
        n_gen          = args.n,
        run_detection  = not args.no_detection,
        run_matrix     = not args.skip_matrix,
        csv_path       = args.csv,
    )
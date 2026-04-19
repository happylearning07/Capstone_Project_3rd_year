"""Local IDS pipeline runner."""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

from gateway.gateway import Gateway
from utils.model_loader import get_model, MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("Pipeline")

FEATURE_COLS_PATH = "saved_state/feature_cols.pkl"

_DEFAULT_ATTACK = {
    "bigan":  "DDoS",
    "aae":    "DDoS",
    "robust": "DDoS",
}

_BINARY_MODELS = {"bigan", "robust"}


def _load_robust_predict():
    """
    Return a predict function for the Robust VAE-BiGAN.
    Uses mu (not reparameterised z) for deterministic, stable scoring -
    identical to comparison.py::predict_robust().
    """
    from models.robust_model import RobustVAEBiGAN

    test_data = np.load("test_data.npz")
    input_dim = test_data["X_test"].shape[1]

    model = RobustVAEBiGAN(input_dim)
    model.load_state_dict(
        torch.load("saved_state/robust_vae_bigan_model.pth", map_location="cpu"))
    model.eval()

    with open("robust_calibration.pkl", "rb") as f:
        calib = pickle.load(f)
    threshold = calib["threshold"]
    flip      = calib["flip"]
    logger.info("Robust model loaded | threshold=%.4f | flip=%s", threshold, flip)

    def predict_fn(X):
        X_t   = torch.FloatTensor(X.astype(np.float32))
        all_s = []
        with torch.no_grad():
            for i in range(0, len(X_t), 1000):
                batch   = X_t[i:i+1000]
                mu, _   = model.encode(batch)
                s       = model.discriminator(
                    torch.cat([batch, mu], dim=1)).squeeze().numpy()
                if s.ndim == 0:
                    s = s.reshape(1)
                all_s.append(s)
        scores = np.concatenate(all_s)
        if flip:
            scores = 1.0 - scores
        preds = (scores > threshold).astype(int)
        return preds, scores

    return predict_fn, input_dim


def run_pipeline(
    mode            = "stream",
    model_type      = "bigan",
    generator_type  = "cgan",
    n_samples       = 100,
    attack_type     = None,
    stage           = "full",
    verbose         = True,
    export_csv      = None,
    epsilon         = None,
    benign_ratio    = 0.70,
    gan_blend_alpha = None,
):
    logger.info("=" * 60)
    logger.info("Pipeline START | mode=%s | model=%s | generator=%s | n=%d",
                mode, model_type, generator_type, n_samples)

    if epsilon is None:
        epsilon = 0.15 if mode == "adversarial" else 0.05

    if attack_type is None and mode in ("attack", "adversarial"):
        attack_type = _DEFAULT_ATTACK.get(model_type, "DDoS")
        logger.info("Auto-selected attack_type=%s for model=%s",
                    attack_type, model_type)

    if model_type == "robust":
        predict_fn, input_dim = _load_robust_predict()
    else:
        predict_fn, input_dim = get_model(model_type)

    if gan_blend_alpha is None:
        gan_blend_alpha = 0.0

    gw = Gateway(
        generator_type  = generator_type,
        gan_blend_alpha = gan_blend_alpha,
        epsilon         = epsilon,
        benign_ratio    = benign_ratio,
    )

    ids_model_obj = None
    if mode == "adversarial":
        if model_type == "bigan":
            from models.bigan_model import BiGAN
            ckpt = torch.load("bigan_final.pth", map_location="cpu")
            ids_model_obj = BiGAN(ckpt["input_dim"], latent_dim=8)
            ids_model_obj.load_state_dict(ckpt["state_dict"])
            ids_model_obj.eval()
            logger.info("BiGAN loaded for FGSM.")

        elif model_type == "aae":
            import os
            from models.aae_model import AAE
            if os.path.exists("aae_final.pth"):
                ckpt = torch.load("aae_final.pth", map_location="cpu")
                ids_model_obj = AAE(input_dim=ckpt["input_dim"],
                                    latent_dim=ckpt.get("latent_dim", 16))
                ids_model_obj.load_state_dict(ckpt["state_dict"])
                logger.info("Full AAE loaded from aae_final.pth.")
            else:
                ckpt = torch.load("encoder_final.pth", map_location="cpu")
                ids_model_obj = AAE(input_dim=ckpt["input_dim"])
                ids_model_obj.encoder.load_state_dict(ckpt["state_dict"])
                logger.warning("aae_final.pth not found - encoder_final.pth used.")
            ids_model_obj.eval()

        elif model_type == "robust":
            from models.robust_model import RobustVAEBiGAN
            test_data = np.load("test_data.npz")
            ids_model_obj = RobustVAEBiGAN(test_data["X_test"].shape[1])
            ids_model_obj.load_state_dict(
                torch.load("saved_state/robust_vae_bigan_model.pth", map_location="cpu"))
            ids_model_obj.eval()
            logger.info("Robust VAE-BiGAN loaded for FGSM.")

    X, y_true, meta = gw.generate(
        mode        = mode,
        n_samples   = n_samples,
        attack_type = attack_type,
        stage       = stage,
        ids_model   = ids_model_obj,
        model_type  = model_type,
    )
    logger.info("Generated %d samples | attacks: %d | benign: %d",
                len(y_true), int(np.sum(y_true > 0)), int(np.sum(y_true == 0)))

    results = _predict_direct(X, y_true, meta, predict_fn, model_type)

    _print_summary(results, mode, model_type, verbose)

    if export_csv:
        save_samples(X, y_true, meta, results, export_csv)

    return results, X, y_true, meta


def _predict_direct(X, y_true, meta, predict_fn, model_type):
    preds, scores = predict_fn(X)
    is_strict_binary = model_type in _BINARY_MODELS
    results = []
    for i in range(len(preds)):
        true_label = int(y_true[i])
        prediction = int(preds[i])

        detected = prediction > 0

        if is_strict_binary:
            correct = (true_label == 0) == (prediction == 0)
        else:
            correct = prediction == true_label

        results.append({
            "sample_id":     i,
            "true_label":    true_label,
            "prediction":    prediction,
            "correct":       correct,
            "detected":      detected,
            "anomaly_score": float(scores[i]),
            "attack_type":   meta[i].get("type", "unknown"),
            "gateway_mode":  meta[i].get("mode", "unknown"),
            "stage":         meta[i].get("stage", "-"),
            "perturbed":     meta[i].get("perturbed", False),
        })
    return results


def _print_summary(results, mode, model_type, verbose):
    total            = len(results)
    is_strict_binary = model_type in _BINARY_MODELS

    n_attacks = sum(1 for r in results if r.get("true_label", 0) > 0)
    n_benign  = total - n_attacks

    attacks_detected = sum(
        1 for r in results
        if r.get("true_label", 0) > 0 and r.get("detected", False)
    )
    missed_attacks = n_attacks - attacks_detected
    false_alarms   = sum(
        1 for r in results
        if r.get("true_label", 0) == 0 and r.get("detected", False)
    )

    detection_rate   = attacks_detected / n_attacks if n_attacks > 0 else 0.0
    false_alarm_rate = false_alarms / n_benign      if n_benign  > 0 else 0.0
    binary_acc       = (attacks_detected + n_benign - false_alarms) / total

    exact_correct = sum(1 for r in results if r.get("correct"))

    wrong_class = 0
    if not is_strict_binary:
        wrong_class = sum(
            1 for r in results
            if r.get("true_label", 0) > 0
            and r.get("detected", False)
            and not r.get("correct")
        )

    print("\n" + "=" * 65)
    print(f"  PIPELINE RESULTS | mode={mode} | model={model_type}")
    print("=" * 65)
    print(f"  Total Samples      : {total}  (attacks={n_attacks}, benign={n_benign})")
    print(f"  -----------------------------------------------------")
    print(f"  Detection Rate     : {detection_rate*100:.1f}%"
          f"  ({attacks_detected}/{n_attacks} attacks flagged)")
    print(f"  False Alarm Rate   : {false_alarm_rate*100:.1f}%"
          f"  ({false_alarms}/{n_benign} benign flagged)")
    print(f"  Accuracy (binary)  : {binary_acc*100:.1f}%")
    print(f"  -----------------------------------------------------")
    print(f"  Missed Attacks     : {missed_attacks}"
          f"  <- attack predicted as benign")
    print(f"  False Alarms       : {false_alarms}"
          f"  <- benign predicted as attack")
    if not is_strict_binary:
        print(f"  Wrong Class        : {wrong_class}"
              f"  <- attack detected but wrong class label")
        print(f"  Exact Correct      : {exact_correct}/{total}"
              f"  ({exact_correct/total*100:.1f}%)  <- multi-class accuracy")

    if model_type == "aae":
        print(f"\n  Note: AAE is a multi-class classifier (5 classes).")
        print(f"  DR/FAR are computed on a binary basis (pred>0 means attack).")
    elif model_type in _BINARY_MODELS:
        model_desc = {
            "bigan":  "BiGAN is a binary anomaly detector.",
            "robust": "Robust VAE-BiGAN is a binary anomaly detector.",
        }.get(model_type, "")
        print(f"\n  Note: {model_desc}")
        print(f"  For per-class labels use --model aae.")

    if mode == "adversarial":
        perturbed   = [r for r in results if r.get("perturbed")]
        n_perturbed = len(perturbed)
        if n_perturbed > 0:
            evasions = sum(
                1 for r in perturbed
                if r.get("true_label", 0) > 0 and not r.get("detected", False)
            )
            survived = n_perturbed - evasions
            print(f"\n  Adversarial Evasion ({n_perturbed} perturbed)")
            print(f"  Evaded IDS (pred=benign)    : "
                  f"{evasions}/{n_perturbed} ({evasions/n_perturbed:.1%})")
            print(f"  IDS held firm (detected)    : "
                  f"{survived}/{n_perturbed} ({survived/n_perturbed:.1%})")

    print("=" * 65)

    if verbose:
        print("\n  Sample-level breakdown (first 20):")
        for r in results[:20]:
            tl       = r.get("true_label", 0)
            pl       = r.get("prediction", 0)
            detected = r.get("detected", False)
            correct  = r.get("correct", False)

            if tl > 0:
                ok = detected
            else:
                ok = not detected
            status = "OK" if ok else "FAIL"

            note = ""
            if tl > 0 and not detected:
                note = "  <- missed"
            elif tl == 0 and detected:
                note = "  <- false alarm"
            elif not is_strict_binary and tl > 0 and detected and not correct:
                note = "  <- wrong class (detected)"

            if mode == "adversarial" and r.get("perturbed"):
                if tl > 0 and not detected:
                    note = "  <- evaded"
                elif tl > 0 and detected:
                    note = "  <- detected"

            print(f"  [{status}] id={r['sample_id']:3d} | "
                  f"true={tl} | pred={pl} | "
                  f"score={r['anomaly_score']:.4f} | "
                  f"type={r.get('attack_type','?'):28s} | "
                  f"perturbed={r.get('perturbed', False)}"
                  f"{note}")
        if total > 20:
            print(f"  ... ({total - 20} more)")


def save_samples(X, y_true, meta, results, filepath):
    try:
        with open(FEATURE_COLS_PATH, "rb") as f:
            feature_cols = pickle.load(f)
    except FileNotFoundError:
        feature_cols = [f"feature_{i}" for i in range(X.shape[1])]

    n_cols = len(feature_cols)
    if X.shape[1] < n_cols:
        pad = np.zeros((len(X), n_cols - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > n_cols:
        X = X[:, :n_cols]

    df = pd.DataFrame(X, columns=feature_cols)
    df["true_label"]   = y_true
    df["attack_type"]  = [m.get("type",      "unknown") for m in meta]
    df["gateway_mode"] = [m.get("mode",      "unknown") for m in meta]
    df["stage"]        = [m.get("stage",     "-")       for m in meta]
    df["perturbed"]    = [m.get("perturbed", False)     for m in meta]

    if results:
        rdf = pd.DataFrame(results)[["sample_id", "prediction",
                                     "correct", "anomaly_score"]]
        df = df.reset_index().rename(columns={"index": "sample_id"})
        df = df.merge(rdf, on="sample_id", how="left").drop(columns=["sample_id"])

    df.to_csv(filepath, index=False)
    logger.info("CSV saved: %s  [%d rows × %d cols]", filepath, len(df), len(df.columns))


def main():
    all_models = list(MODEL_REGISTRY.keys()) + ["robust"]

    parser = argparse.ArgumentParser(
        description="IoT IDS Pipeline - GMM + GAN gateway edition")

    parser.add_argument("--mode",      choices=["stream", "attack", "adversarial"],
                        default="stream")
    parser.add_argument("--model",     choices=all_models, default="bigan")
    parser.add_argument("--generator", choices=["cgan", "tvae", "none"],
                        default="cgan",
                        help="Traffic refiner: cgan/tvae, or 'none' for GMM-only.")
    parser.add_argument("--n",         type=int, default=100)
    parser.add_argument("--attack",    type=str, default=None,
                        help="Attack class. Default: DDoS for all models. "
                             "Matches comparison.py stress_attacks list.")
    parser.add_argument("--stage",     choices=["early", "full"], default="full")
    parser.add_argument("--epsilon",   type=float, default=None,
                        help="FGSM epsilon. Default: 0.05 (stream/attack), "
                             "0.15 (adversarial)")
    parser.add_argument("--benign-ratio", type=float, default=0.70)
    parser.add_argument("--gan-blend",    type=float, default=None,
                        help="GAN blend alpha (default: 0.0, matching comparison.py).")
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--export-csv", type=str, default=None)
    parser.add_argument("--save",      type=str, default=None)

    args = parser.parse_args()

    extra_kwargs = {}
    if args.gan_blend is not None:
        extra_kwargs["gan_blend_alpha"] = args.gan_blend

    results, X, y_true, meta = run_pipeline(
        mode            = args.mode,
        model_type      = args.model,
        generator_type  = args.generator,
        n_samples       = args.n,
        attack_type     = args.attack,
        stage           = args.stage,
        verbose         = not args.quiet,
        export_csv      = args.export_csv,
        epsilon         = args.epsilon,
        benign_ratio    = args.benign_ratio,
        **extra_kwargs,
    )

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results JSON saved to {args.save}")


if __name__ == "__main__":
    main()
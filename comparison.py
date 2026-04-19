"""Compare AAE, BiGAN, and Robust VAE-BiGAN."""

import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

from models.bigan_model  import BiGAN
from models.robust_model import RobustVAEBiGAN
from models.aae_model    import AAE
from gateway.gateway     import Gateway

def load_all_models(input_dim):
    bigan   = BiGAN(input_dim)
    bg_ckpt = torch.load("bigan_final.pth", map_location="cpu")
    bigan.load_state_dict(
        bg_ckpt["state_dict"] if "state_dict" in bg_ckpt else bg_ckpt)
    bigan.eval()
    with open("bigan_calibration.pkl", "rb") as f:
        bg_calib = pickle.load(f)

    robust = RobustVAEBiGAN(input_dim)
    robust.load_state_dict(
        torch.load("saved_state/robust_vae_bigan_model.pth", map_location="cpu"))
    robust.eval()
    try:
        with open("robust_calibration.pkl", "rb") as f:
            rob_calib = pickle.load(f)
        print(f"  Robust calibration: threshold={rob_calib['threshold']:.4f}  "
              f"flip={rob_calib['flip']}")
    except FileNotFoundError:
        raise FileNotFoundError(
            "robust_calibration.pkl not found. Run train_robust_bigan.py first.")

    aae      = AAE(input_dim=input_dim)
    aae_ckpt = torch.load("aae_final.pth", map_location="cpu")
    aae.load_state_dict(aae_ckpt["state_dict"])
    aae.eval()
    with open("aae_classifier.pkl", "rb") as f:
        aae_rf = pickle.load(f)

    return (bigan, bg_calib), (robust, rob_calib), (aae, aae_rf)


def predict_bigan(model_data, X):
    model, calib = model_data
    X_t = torch.FloatTensor(X.astype(np.float32))
    with torch.no_grad():
        z       = model.encoder(X_t)
        recon   = torch.mean((X_t - model.generator(z)) ** 2, dim=1).numpy()
        d_score = model.discriminator(
            torch.cat([X_t, z], dim=1)).squeeze().numpy()
        if d_score.ndim == 0:
            d_score = d_score.reshape(1)
    span       = calib["r_max"] - calib["r_min"]
    recon_norm = np.clip((recon - calib["r_min"]) / (span + 1e-8), 0.0, 1.0)
    raw        = 0.5 * (1.0 - d_score) + 0.5 * recon_norm
    scores     = (1.0 - raw) if calib["flip"] else raw
    return (scores > calib["threshold"]).astype(int), scores


def predict_robust(model_data, X):
    model, calib = model_data
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
    if calib["flip"]:
        scores = 1.0 - scores
    return (scores > calib["threshold"]).astype(int), scores


def predict_aae(model_data, X):
    aae, rf = model_data
    X_t = torch.FloatTensor(X.astype(np.float32))
    with torch.no_grad():
        Z = aae.encoder(X_t).numpy()
    mc_preds = rf.predict(Z)
    probas   = rf.predict_proba(Z).max(axis=1)
    return (mc_preds > 0).astype(int), probas


def detection_rate(y_true_bin, y_pred):
    attacks = y_true_bin == 1
    if attacks.sum() == 0:
        return 0.0
    return float(y_pred[attacks].mean()) * 100


def false_alarm_rate(y_true_bin, y_pred):
    benign = y_true_bin == 0
    if benign.sum() == 0:
        return 0.0
    return float(y_pred[benign].mean()) * 100


def fmt(y_true_bin, y_pred):
    """One-liner: Acc / DR / FAR."""
    acc = accuracy_score(y_true_bin, y_pred) * 100
    dr  = detection_rate(y_true_bin, y_pred)
    far = false_alarm_rate(y_true_bin, y_pred)
    return f"Acc={acc:5.1f}%  DR={dr:5.1f}%  FAR={far:4.1f}%"


def fmt_honest(y_true_bin, y_pred):
    """Show TP/TN/FP/FN counts."""
    attacks = y_true_bin == 1
    benign  = y_true_bin == 0
    tp = int(y_pred[attacks].sum())
    tn = int((y_pred[benign] == 0).sum())
    fp = int(y_pred[benign].sum())
    fn = int((y_pred[attacks] == 0).sum())
    dr  = tp / max(attacks.sum(), 1) * 100
    far = fp / max(benign.sum(),  1) * 100
    return f"DR={dr:5.1f}%  FAR={far:4.1f}%  TP={tp}  TN={tn}  FP={fp}  FN={fn}"


if __name__ == "__main__":

    print("Loading test data and models...")
    test_data  = np.load("test_data.npz")
    X_db, y_db = test_data["X_test"], test_data["y_test"]
    y_db_bin   = (y_db > 0).astype(int)
    input_dim  = X_db.shape[1]

    n_benign_db = int((y_db_bin == 0).sum())
    n_attack_db = int((y_db_bin == 1).sum())
    print(f"  input_dim={input_dim}  test_samples={len(X_db)}")
    print(f"  Benign={n_benign_db}  Attack={n_attack_db}  "
          f"(ratio 1:{n_attack_db // max(n_benign_db, 1)})\n")

    bigan_data, robust_data, aae_data = load_all_models(input_dim)
    bigan_model  = bigan_data[0]
    robust_model = robust_data[0]
    aae_model    = aae_data[0]
    print("  All models loaded.\n")

    gateway = Gateway(epsilon=0.15, gan_blend_alpha=0.0)

    W   = 130      # total line width
    COL = 38       # column width per model result

    def header_row():
        return (f"  {'SCENARIO':<46}| "
                f"{'AAE + RF':<{COL}}| "
                f"{'BiGAN':<{COL}}| "
                f"Robust VAE-BiGAN")

    def divider():
        print("  " + "-" * (W - 2))

    print("=" * W)
    print(f"{'COMPARISON - AAE vs BiGAN vs Robust VAE-BiGAN':^{W}}")
    print("=" * W)
    print(f"  DR = Detection Rate   FAR = False Alarm Rate   "
          f"TP/TN/FP/FN = sample counts\n")

    print(f"  [1] BASELINE - Real Test Split")
    print(f"  {n_benign_db} benign vs {n_attack_db} attacks "
          f"({n_attack_db / len(X_db) * 100:.1f}% attacks).")
    print()

    p_aae, _ = predict_aae(aae_data,    X_db)
    p_bg,  _ = predict_bigan(bigan_data,  X_db)
    p_rob, _ = predict_robust(robust_data, X_db)

    print(f"  {'Model':<22} Result")
    divider()
    print(f"  {'AAE + RF':<22} {fmt_honest(y_db_bin, p_aae)}")
    print(f"  {'BiGAN':<22} {fmt_honest(y_db_bin, p_bg)}")
    print(f"  {'Robust VAE-BiGAN':<22} {fmt_honest(y_db_bin, p_rob)}")
    print()
    print(f"  AAE note: C&C-HeartBeat has low recall in some runs - only 11 test")
    print(f"  samples, classifier cannot generalise from that alone.")
    print("-" * W)

    print(f"\n  [2] STREAM MODE - 70% benign / 30% attacks")
    print(header_row())
    divider()

    for n in [200, 500]:
        X_s, y_s, _ = gateway.generate(mode="stream", n_samples=n)
        y_s_bin      = (y_s > 0).astype(int)
        p_aae, _ = predict_aae(aae_data,    X_s)
        p_bg,  _ = predict_bigan(bigan_data,  X_s)
        p_rob, _ = predict_robust(robust_data, X_s)
        lbl = f"  {'Stream (n='+str(n)+')':<44}"
        print(f"{lbl}| {fmt(y_s_bin, p_aae):<{COL}}| "
              f"{fmt(y_s_bin, p_bg):<{COL}}| "
              f"{fmt(y_s_bin, p_rob)}")

    print("-" * W)

    print(f"\n  [3] ADVERSARIAL EVASION - White-box FGSM ε=0.15")
    print(header_row())
    divider()

    stress_attacks = ["DDoS", "Okiru", "PartOfAHorizontalPortScan"]

    for atk in stress_attacks:
        X_adv_bg, y_adv, _ = gateway.generate(
            mode="adversarial", n_samples=300, attack_type=atk,
            ids_model=bigan_model,  model_type="bigan")
        X_adv_aae, _, _ = gateway.generate(
            mode="adversarial", n_samples=300, attack_type=atk,
            ids_model=aae_model,   model_type="aae")
        X_adv_rob, _, _ = gateway.generate(
            mode="adversarial", n_samples=300, attack_type=atk,
            ids_model=robust_model, model_type="robust")

        y_adv_bin = (y_adv > 0).astype(int)
        p_aae, _ = predict_aae(aae_data,    X_adv_aae)
        p_bg,  _ = predict_bigan(bigan_data,  X_adv_bg)
        p_rob, _ = predict_robust(robust_data, X_adv_rob)

        lbl = f"  {'FGSM ε=0.15 ('+atk+')':<44}"
        print(f"{lbl}| {fmt(y_adv_bin, p_aae):<{COL}}| "
              f"{fmt(y_adv_bin, p_bg):<{COL}}| "
              f"{fmt(y_adv_bin, p_rob)}")

    print("-" * W)
    print("=" * W)
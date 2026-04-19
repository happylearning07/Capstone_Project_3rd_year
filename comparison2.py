# # """Compare AAE, BiGAN, Robust VAE-BiGAN, and VAE-BiGAN + ANN (new)."""

# # import torch
# # import numpy as np
# # import pickle
# # from sklearn.metrics import accuracy_score

# # from models.bigan_model  import BiGAN
# # from models.robust_model import RobustVAEBiGAN
# # from models.aae_model    import AAE
# # from gateway.gateway     import Gateway

# # # ── VAE-BiGAN + ANN architecture (mirrors train_vae_bigan_ann.py) ─────────────

# # import torch.nn as nn

# # class _Encoder(nn.Module):
# #     def __init__(self, input_dim: int, latent_dim: int):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
# #             nn.Linear(128, 64),        nn.LeakyReLU(0.2), nn.BatchNorm1d(64),
# #         )
# #         self.fc_mu     = nn.Linear(64, latent_dim)
# #         self.fc_logvar = nn.Linear(64, latent_dim)

# #     def forward(self, x):
# #         h = self.net(x)
# #         return self.fc_mu(h), self.fc_logvar(h)


# # class _ANNClassifier(nn.Module):
# #     def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.3):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(latent_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
# #             nn.Linear(64, 32),         nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout),
# #             nn.Linear(32, num_classes),
# #         )

# #     def forward(self, z):
# #         return self.net(z)


# # # ── Model loaders ──────────────────────────────────────────────────────────────

# # def load_all_models(input_dim):
# #     # BiGAN
# #     bigan   = BiGAN(input_dim)
# #     bg_ckpt = torch.load("bigan_final.pth", map_location="cpu")
# #     bigan.load_state_dict(
# #         bg_ckpt["state_dict"] if "state_dict" in bg_ckpt else bg_ckpt)
# #     bigan.eval()
# #     with open("bigan_calibration.pkl", "rb") as f:
# #         bg_calib = pickle.load(f)

# #     # Robust VAE-BiGAN
# #     robust = RobustVAEBiGAN(input_dim)
# #     robust.load_state_dict(
# #         torch.load("saved_state/robust_vae_bigan_model.pth", map_location="cpu"))
# #     robust.eval()
# #     with open("robust_calibration.pkl", "rb") as f:
# #         rob_calib = pickle.load(f)
# #     print(f"  Robust calibration: threshold={rob_calib['threshold']:.4f}  "
# #           f"flip={rob_calib['flip']}")

# #     # AAE + RF
# #     aae      = AAE(input_dim=input_dim)
# #     aae_ckpt = torch.load("aae_final.pth", map_location="cpu")
# #     aae.load_state_dict(aae_ckpt["state_dict"])
# #     aae.eval()
# #     with open("aae_classifier.pkl", "rb") as f:
# #         aae_rf = pickle.load(f)

# #     # VAE-BiGAN + ANN  (new model)
# #     vae_bigan_ann = _load_vae_bigan_ann(input_dim)

# #     return (bigan, bg_calib), (robust, rob_calib), (aae, aae_rf), vae_bigan_ann


# # def _load_vae_bigan_ann(input_dim):
# #     """
# #     Load the VAE-BiGAN encoder + ANN classifier saved by train_vae_bigan_ann.py.
# #     Returns None if the checkpoint files don't exist yet.
# #     """
# #     model_path = "saved_state/vae_bigan_ann_model.pth"
# #     clf_path   = "saved_state/ann_classifier.pth"
# #     meta_path  = "saved_state/ann_multiclass_meta.pkl"

# #     for p in [model_path, clf_path, meta_path]:
# #         if not __import__("os").path.exists(p):
# #             print(f"  [WARN] VAE-BiGAN+ANN: '{p}' not found — model will be skipped.")
# #             print(f"         Run: python train_vae_bigan_ann.py")
# #             return None

# #     # Load meta
# #     with open(meta_path, "rb") as f:
# #         meta = pickle.load(f)
# #     latent_dim  = meta["latent_dim"]
# #     num_classes = meta["num_classes"]
# #     idx_to_name = meta["idx_to_name"]
# #     code_to_idx = meta["code_to_idx"]

# #     # Reconstruct encoder
# #     enc_ckpt = torch.load(model_path, map_location="cpu")
# #     encoder  = _Encoder(input_dim, latent_dim)
# #     encoder.load_state_dict(enc_ckpt["encoder"])
# #     encoder.eval()

# #     # Reconstruct classifier
# #     clf_ckpt   = torch.load(clf_path, map_location="cpu")
# #     classifier = _ANNClassifier(latent_dim, num_classes)
# #     classifier.load_state_dict(clf_ckpt["classifier"])
# #     classifier.eval()

# #     print(f"  VAE-BiGAN+ANN: latent_dim={latent_dim}  num_classes={num_classes}")
# #     return encoder, classifier, meta


# # # ── Prediction functions ───────────────────────────────────────────────────────

# # def predict_bigan(model_data, X):
# #     model, calib = model_data
# #     X_t = torch.FloatTensor(X.astype(np.float32))
# #     with torch.no_grad():
# #         z       = model.encoder(X_t)
# #         recon   = torch.mean((X_t - model.generator(z)) ** 2, dim=1).numpy()
# #         d_score = model.discriminator(
# #             torch.cat([X_t, z], dim=1)).squeeze().numpy()
# #         if d_score.ndim == 0:
# #             d_score = d_score.reshape(1)
# #     span       = calib["r_max"] - calib["r_min"]
# #     recon_norm = np.clip((recon - calib["r_min"]) / (span + 1e-8), 0.0, 1.0)
# #     raw        = 0.5 * (1.0 - d_score) + 0.5 * recon_norm
# #     scores     = (1.0 - raw) if calib["flip"] else raw
# #     return (scores > calib["threshold"]).astype(int), scores


# # def predict_robust(model_data, X):
# #     model, calib = model_data
# #     X_t   = torch.FloatTensor(X.astype(np.float32))
# #     all_s = []
# #     with torch.no_grad():
# #         for i in range(0, len(X_t), 1000):
# #             batch   = X_t[i:i+1000]
# #             mu, _   = model.encode(batch)
# #             s       = model.discriminator(
# #                 torch.cat([batch, mu], dim=1)).squeeze().numpy()
# #             if s.ndim == 0:
# #                 s = s.reshape(1)
# #             all_s.append(s)
# #     scores = np.concatenate(all_s)
# #     if calib["flip"]:
# #         scores = 1.0 - scores
# #     return (scores > calib["threshold"]).astype(int), scores


# # def predict_aae(model_data, X):
# #     aae, rf = model_data
# #     X_t = torch.FloatTensor(X.astype(np.float32))
# #     with torch.no_grad():
# #         Z = aae.encoder(X_t).numpy()
# #     mc_preds = rf.predict(Z)
# #     probas   = rf.predict_proba(Z).max(axis=1)
# #     return (mc_preds > 0).astype(int), probas


# # def predict_vae_bigan_ann(model_data, X):
# #     """
# #     VAE-BiGAN + ANN prediction.

# #     Returns binary preds (0=benign, 1=attack) and per-sample confidence scores.
# #     The multiclass index 0 is assumed to be Benign (code 0 in reverse_label_map).
# #     Any prediction with class index > 0 is treated as an attack.
# #     """
# #     if model_data is None:
# #         n = len(X)
# #         return np.zeros(n, dtype=int), np.zeros(n, dtype=np.float32)

# #     encoder, classifier, meta = model_data
# #     code_to_idx = meta["code_to_idx"]

# #     # The benign pipeline code is 0; find its consecutive index
# #     benign_idx = code_to_idx.get(0, 0)

# #     X_t = torch.FloatTensor(X.astype(np.float32))
# #     with torch.no_grad():
# #         mu, _ = encoder(X_t)
# #         logits = classifier(mu)
# #         probs  = torch.softmax(logits, dim=1)
# #         preds  = logits.argmax(dim=1).numpy()

# #     # Confidence score = 1 - P(benign)  so higher = more anomalous
# #     scores = 1.0 - probs[:, benign_idx].numpy()
# #     binary = (preds != benign_idx).astype(int)
# #     return binary, scores


# # # ── Metric helpers ─────────────────────────────────────────────────────────────

# # def detection_rate(y_true_bin, y_pred):
# #     attacks = y_true_bin == 1
# #     if attacks.sum() == 0:
# #         return 0.0
# #     return float(y_pred[attacks].mean()) * 100


# # def false_alarm_rate(y_true_bin, y_pred):
# #     benign = y_true_bin == 0
# #     if benign.sum() == 0:
# #         return 0.0
# #     return float(y_pred[benign].mean()) * 100


# # def fmt(y_true_bin, y_pred):
# #     acc = accuracy_score(y_true_bin, y_pred) * 100
# #     dr  = detection_rate(y_true_bin, y_pred)
# #     far = false_alarm_rate(y_true_bin, y_pred)
# #     return f"Acc={acc:5.1f}%  DR={dr:5.1f}%  FAR={far:4.1f}%"


# # def fmt_honest(y_true_bin, y_pred):
# #     attacks = y_true_bin == 1
# #     benign  = y_true_bin == 0
# #     tp = int(y_pred[attacks].sum())
# #     tn = int((y_pred[benign] == 0).sum())
# #     fp = int(y_pred[benign].sum())
# #     fn = int((y_pred[attacks] == 0).sum())
# #     dr  = tp / max(attacks.sum(), 1) * 100
# #     far = fp / max(benign.sum(),  1) * 100
# #     return f"DR={dr:5.1f}%  FAR={far:4.1f}%  TP={tp}  TN={tn}  FP={fp}  FN={fn}"


# # # ── Multi-class breakdown (new model only) ─────────────────────────────────────

# # def multiclass_report(model_data, X, y_true_codes, code_to_name):
# #     """Print per-class accuracy for VAE-BiGAN+ANN using its internal class indices."""
# #     if model_data is None:
# #         print("  VAE-BiGAN+ANN not loaded — skipping multiclass report.")
# #         return

# #     encoder, classifier, meta = model_data
# #     code_to_idx = meta["code_to_idx"]
# #     idx_to_name = meta["idx_to_name"]

# #     X_t = torch.FloatTensor(X.astype(np.float32))
# #     with torch.no_grad():
# #         mu, _ = encoder(X_t)
# #         preds  = classifier(mu).argmax(dim=1).numpy()

# #     # Map true pipeline codes -> consecutive indices (skip unknown codes)
# #     true_idx = np.array([code_to_idx.get(int(c), -1) for c in y_true_codes])
# #     valid    = true_idx >= 0

# #     from sklearn.metrics import classification_report as skl_report
# #     class_names = [idx_to_name[i] for i in range(meta["num_classes"])]
# #     print(skl_report(true_idx[valid], preds[valid],
# #                      target_names=class_names, zero_division=0))


# # # ── Main ───────────────────────────────────────────────────────────────────────

# # if __name__ == "__main__":

# #     print("Loading test data and models...")
# #     test_data  = np.load("test_data.npz")
# #     X_db, y_db = test_data["X_test"], test_data["y_test"]
# #     y_db_bin   = (y_db > 0).astype(int)
# #     input_dim  = X_db.shape[1]

# #     n_benign_db = int((y_db_bin == 0).sum())
# #     n_attack_db = int((y_db_bin == 1).sum())
# #     print(f"  input_dim={input_dim}  test_samples={len(X_db)}")
# #     print(f"  Benign={n_benign_db}  Attack={n_attack_db}  "
# #           f"(ratio 1:{n_attack_db // max(n_benign_db, 1)})\n")

# #     bigan_data, robust_data, aae_data, vae_ann_data = load_all_models(input_dim)
# #     bigan_model  = bigan_data[0]
# #     robust_model = robust_data[0]
# #     aae_model    = aae_data[0]
# #     print("  All models loaded.\n")

# #     # Load reverse label map for display
# #     try:
# #         with open("saved_state/reverse_label_map.pkl", "rb") as f:
# #             rev_map = pickle.load(f)
# #         code_to_name = {v: k for k, v in rev_map.items()}
# #     except FileNotFoundError:
# #         code_to_name = {}

# #     gateway = Gateway(epsilon=0.15, gan_blend_alpha=0.0)

# #     W   = 160
# #     COL = 38

# #     def header_row():
# #         return (f"  {'SCENARIO':<46}| "
# #                 f"{'AAE + RF':<{COL}}| "
# #                 f"{'BiGAN':<{COL}}| "
# #                 f"{'Robust VAE-BiGAN':<{COL}}| "
# #                 f"VAE-BiGAN + ANN (new)")

# #     def divider():
# #         print("  " + "-" * (W - 2))

# #     print("=" * W)
# #     print(f"{'COMPARISON - AAE vs BiGAN vs Robust VAE-BiGAN vs VAE-BiGAN+ANN':^{W}}")
# #     print("=" * W)
# #     print(f"  DR = Detection Rate   FAR = False Alarm Rate   "
# #           f"TP/TN/FP/FN = sample counts\n")

# #     # ── [1] Baseline ──────────────────────────────────────────────────────────
# #     print(f"  [1] BASELINE - Real Test Split")
# #     print(f"  {n_benign_db} benign vs {n_attack_db} attacks "
# #           f"({n_attack_db / len(X_db) * 100:.1f}% attacks).")
# #     print()

# #     p_aae,     _ = predict_aae(aae_data,         X_db)
# #     p_bg,      _ = predict_bigan(bigan_data,      X_db)
# #     p_rob,     _ = predict_robust(robust_data,    X_db)
# #     p_vae_ann, _ = predict_vae_bigan_ann(vae_ann_data, X_db)

# #     print(f"  {'Model':<25} Result")
# #     divider()
# #     print(f"  {'AAE + RF':<25} {fmt_honest(y_db_bin, p_aae)}")
# #     print(f"  {'BiGAN':<25} {fmt_honest(y_db_bin, p_bg)}")
# #     print(f"  {'Robust VAE-BiGAN':<25} {fmt_honest(y_db_bin, p_rob)}")
# #     print(f"  {'VAE-BiGAN + ANN':<25} {fmt_honest(y_db_bin, p_vae_ann)}")
# #     print()
# #     print(f"  Notes:")
# #     print(f"    AAE: C&C-HeartBeat has low recall — only 11 test samples.")
# #     print(f"    VAE-BiGAN+ANN: multiclass classifier; binary DR treats any non-benign pred as attack.")

# #     # ── [1b] VAE-BiGAN+ANN multiclass detail ──────────────────────────────────
# #     if vae_ann_data is not None:
# #         print(f"\n  [1b] VAE-BiGAN+ANN - Multiclass Classification Report (real test set)")
# #         divider()
# #         multiclass_report(vae_ann_data, X_db, y_db, code_to_name)

# #     print("-" * W)

# #     # ── [2] Stream mode ───────────────────────────────────────────────────────
# #     print(f"\n  [2] STREAM MODE - 70% benign / 30% attacks")
# #     print(header_row())
# #     divider()

# #     for n in [200, 500]:
# #         X_s, y_s, _ = gateway.generate(mode="stream", n_samples=n)
# #         y_s_bin      = (y_s > 0).astype(int)
# #         p_aae,     _ = predict_aae(aae_data,         X_s)
# #         p_bg,      _ = predict_bigan(bigan_data,      X_s)
# #         p_rob,     _ = predict_robust(robust_data,    X_s)
# #         p_vae_ann, _ = predict_vae_bigan_ann(vae_ann_data, X_s)
# #         lbl = f"  {'Stream (n='+str(n)+')':<44}"
# #         print(f"{lbl}| {fmt(y_s_bin, p_aae):<{COL}}| "
# #               f"{fmt(y_s_bin, p_bg):<{COL}}| "
# #               f"{fmt(y_s_bin, p_rob):<{COL}}| "
# #               f"{fmt(y_s_bin, p_vae_ann)}")

# #     print("-" * W)

# #     # ── [3] Adversarial evasion ───────────────────────────────────────────────
# #     print(f"\n  [3] ADVERSARIAL EVASION - White-box FGSM ε=0.15")
# #     print(header_row())
# #     divider()

# #     stress_attacks = ["DDoS", "Okiru", "PartOfAHorizontalPortScan"]

# #     for atk in stress_attacks:
# #         X_adv_bg,      y_adv, _ = gateway.generate(
# #             mode="adversarial", n_samples=300, attack_type=atk,
# #             ids_model=bigan_model,  model_type="bigan")
# #         X_adv_aae,         _, _ = gateway.generate(
# #             mode="adversarial", n_samples=300, attack_type=atk,
# #             ids_model=aae_model,   model_type="aae")
# #         X_adv_rob,         _, _ = gateway.generate(
# #             mode="adversarial", n_samples=300, attack_type=atk,
# #             ids_model=robust_model, model_type="robust")

# #         # For VAE-BiGAN+ANN there is no dedicated FGSM path in gateway yet,
# #         # so we test its robustness against the BiGAN-optimised perturbation
# #         # (transfer attack) — a realistic black-box threat scenario.
# #         X_adv_vaa = X_adv_bg

# #         y_adv_bin  = (y_adv > 0).astype(int)
# #         p_aae,     _ = predict_aae(aae_data,              X_adv_aae)
# #         p_bg,      _ = predict_bigan(bigan_data,           X_adv_bg)
# #         p_rob,     _ = predict_robust(robust_data,         X_adv_rob)
# #         p_vae_ann, _ = predict_vae_bigan_ann(vae_ann_data, X_adv_vaa)

# #         lbl = f"  {'FGSM ε=0.15 ('+atk+')':<44}"
# #         print(f"{lbl}| {fmt(y_adv_bin, p_aae):<{COL}}| "
# #               f"{fmt(y_adv_bin, p_bg):<{COL}}| "
# #               f"{fmt(y_adv_bin, p_rob):<{COL}}| "
# #               f"{fmt(y_adv_bin, p_vae_ann)}")

# #     print("-" * W)
# #     print(f"\n  * VAE-BiGAN+ANN adversarial column uses BiGAN-optimised perturbations")
# #     print(f"    (transfer / black-box attack) — no dedicated FGSM path implemented yet.")
# #     print("=" * W)























































































































# """
# comparison2.py
# --------------
# Compare AAE, BiGAN, Robust VAE-BiGAN, and One-Class SVM (OC-SVM).

# OC-SVM is added following:
#   "Mitigating IoT botnet attacks: An early-stage explainable network-based
#    anomaly detection approach" (Amara Korba et al., Computer Communications 2025).

# That paper reports OC-SVM as the top-performing anomaly detector, achieving
# 99.99% recall and 1.53% FPR on packet-level C&C traffic.

# Run after training all models:
#     python train_aae.py
#     python train_aae_classifier.py
#     python train_bigan.py
#     python train_robust.py   (then calibrate_robust.py)
#     python train_ocsvm.py    ← new
#     python comparison2.py
# """

# import torch
# import numpy as np
# import pickle

# from sklearn.metrics import accuracy_score

# from models.bigan_model  import BiGAN
# from models.robust_model import RobustVAEBiGAN
# from models.aae_model    import AAE
# from models.ocsvm_model  import OCSVMDetector   # ← new
# from gateway.gateway     import Gateway


# # ---------------------------------------------------------------------------
# # Model loading helpers
# # ---------------------------------------------------------------------------

# def load_all_models(input_dim):
#     """Load BiGAN, Robust VAE-BiGAN, AAE, and OC-SVM."""

#     # -- BiGAN ---------------------------------------------------------------
#     bigan   = BiGAN(input_dim)
#     bg_ckpt = torch.load("bigan_final.pth", map_location="cpu")
#     bigan.load_state_dict(
#         bg_ckpt["state_dict"] if "state_dict" in bg_ckpt else bg_ckpt)
#     bigan.eval()

#     with open("bigan_calibration.pkl", "rb") as f:
#         bg_calib = pickle.load(f)

#     # -- Robust VAE-BiGAN ----------------------------------------------------
#     robust = RobustVAEBiGAN(input_dim)
#     robust.load_state_dict(
#         torch.load("saved_state/robust_vae_bigan_model.pth", map_location="cpu"))
#     robust.eval()

#     try:
#         with open("robust_calibration.pkl", "rb") as f:
#             rob_calib = pickle.load(f)
#         print(f"  Robust calibration: threshold={rob_calib['threshold']:.4f}  "
#               f"flip={rob_calib['flip']}")
#     except FileNotFoundError:
#         raise FileNotFoundError(
#             "robust_calibration.pkl not found. "
#             "Run calibrate_robust.py (or train_robust.py) first.")

#     # -- AAE + RF ------------------------------------------------------------
#     aae      = AAE(input_dim=input_dim)
#     aae_ckpt = torch.load("aae_final.pth", map_location="cpu")
#     aae.load_state_dict(aae_ckpt["state_dict"])
#     aae.eval()

#     with open("aae_classifier.pkl", "rb") as f:
#         aae_rf = pickle.load(f)

#     # -- One-Class SVM -------------------------------------------------------
#     try:
#         ocsvm    = OCSVMDetector.load("ocsvm_model.pkl")
#         with open("ocsvm_calibration.pkl", "rb") as f:
#             ocsvm_calib = pickle.load(f)
#         print(f"  OC-SVM calibration: threshold={ocsvm_calib['threshold']:.4f}  "
#               f"nu={ocsvm_calib['nu']}  gamma={ocsvm_calib['gamma']}")
#     except FileNotFoundError:
#         raise FileNotFoundError(
#             "ocsvm_model.pkl or ocsvm_calibration.pkl not found. "
#             "Run train_ocsvm.py first.")

#     return (
#         (bigan,  bg_calib),
#         (robust, rob_calib),
#         (aae,    aae_rf),
#         (ocsvm,  ocsvm_calib),
#     )


# # ---------------------------------------------------------------------------
# # Prediction helpers
# # ---------------------------------------------------------------------------

# def predict_bigan(model_data, X):
#     model, calib = model_data
#     X_t = torch.FloatTensor(X.astype(np.float32))
#     with torch.no_grad():
#         z       = model.encoder(X_t)
#         recon   = torch.mean((X_t - model.generator(z)) ** 2, dim=1).numpy()
#         d_score = model.discriminator(
#             torch.cat([X_t, z], dim=1)).squeeze().numpy()
#         if d_score.ndim == 0:
#             d_score = d_score.reshape(1)

#     span       = calib["r_max"] - calib["r_min"]
#     recon_norm = np.clip((recon - calib["r_min"]) / (span + 1e-8), 0.0, 1.0)
#     raw        = 0.5 * (1.0 - d_score) + 0.5 * recon_norm
#     scores     = (1.0 - raw) if calib["flip"] else raw
#     return (scores > calib["threshold"]).astype(int), scores


# def predict_robust(model_data, X):
#     model, calib = model_data
#     X_t   = torch.FloatTensor(X.astype(np.float32))
#     all_s = []
#     with torch.no_grad():
#         for i in range(0, len(X_t), 1000):
#             batch = X_t[i:i+1000]
#             mu, _ = model.encode(batch)
#             s     = model.discriminator(
#                 torch.cat([batch, mu], dim=1)).squeeze().numpy()
#             if s.ndim == 0:
#                 s = s.reshape(1)
#             all_s.append(s)
#     scores = np.concatenate(all_s)
#     if calib["flip"]:
#         scores = 1.0 - scores
#     return (scores > calib["threshold"]).astype(int), scores


# def predict_aae(model_data, X):
#     aae, rf = model_data
#     X_t = torch.FloatTensor(X.astype(np.float32))
#     with torch.no_grad():
#         Z = aae.encoder(X_t).numpy()
#     mc_preds = rf.predict(Z)
#     probas   = rf.predict_proba(Z).max(axis=1)
#     return (mc_preds > 0).astype(int), probas


# def predict_ocsvm(model_data, X):
#     """
#     One-Class SVM prediction.

#     model_data = (OCSVMDetector, calib_dict)
#     Returns (preds, anomaly_scores) with preds in {0, 1}.
#     """
#     detector, _calib = model_data
#     preds, scores = detector.predict(X.astype(np.float32))
#     return preds, scores


# # ---------------------------------------------------------------------------
# # Metric helpers
# # ---------------------------------------------------------------------------

# def detection_rate(y_true_bin, y_pred):
#     attacks = y_true_bin == 1
#     if attacks.sum() == 0:
#         return 0.0
#     return float(y_pred[attacks].mean()) * 100


# def false_alarm_rate(y_true_bin, y_pred):
#     benign = y_true_bin == 0
#     if benign.sum() == 0:
#         return 0.0
#     return float(y_pred[benign].mean()) * 100


# def fmt(y_true_bin, y_pred):
#     """One-liner: Acc / DR / FAR."""
#     acc = accuracy_score(y_true_bin, y_pred) * 100
#     dr  = detection_rate(y_true_bin, y_pred)
#     far = false_alarm_rate(y_true_bin, y_pred)
#     return f"Acc={acc:5.1f}%  DR={dr:5.1f}%  FAR={far:4.1f}%"


# def fmt_honest(y_true_bin, y_pred):
#     """Show TP/TN/FP/FN counts."""
#     attacks = y_true_bin == 1
#     benign  = y_true_bin == 0
#     tp = int(y_pred[attacks].sum())
#     tn = int((y_pred[benign] == 0).sum())
#     fp = int(y_pred[benign].sum())
#     fn = int((y_pred[attacks] == 0).sum())
#     dr  = tp / max(attacks.sum(), 1) * 100
#     far = fp / max(benign.sum(),  1) * 100
#     return f"DR={dr:5.1f}%  FAR={far:4.1f}%  TP={tp}  TN={tn}  FP={fp}  FN={fn}"


# # ---------------------------------------------------------------------------
# # Main comparison
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":

#     print("Loading test data and models...")

#     test_data  = np.load("test_data.npz")
#     X_db, y_db = test_data["X_test"], test_data["y_test"]
#     y_db_bin   = (y_db > 0).astype(int)
#     input_dim  = X_db.shape[1]

#     n_benign_db = int((y_db_bin == 0).sum())
#     n_attack_db = int((y_db_bin == 1).sum())

#     print(f"  input_dim={input_dim}  test_samples={len(X_db)}")
#     print(f"  Benign={n_benign_db}  Attack={n_attack_db}  "
#           f"(ratio 1:{n_attack_db // max(n_benign_db, 1)})\n")

#     bigan_data, robust_data, aae_data, ocsvm_data = load_all_models(input_dim)

#     bigan_model  = bigan_data[0]
#     robust_model = robust_data[0]
#     aae_model    = aae_data[0]
#     # OC-SVM does not need a separate model object for Gateway FGSM
#     # (sklearn models are not differentiable), so adversarial mode
#     # for OC-SVM uses BiGAN's FGSM perturbations for a fair comparison.

#     print("  All models loaded.\n")

#     gateway = Gateway(epsilon=0.15, gan_blend_alpha=0.0)

#     W   = 160          # total line width
#     COL = 38           # column width per model result

#     def header_row():
#         return (f"  {'SCENARIO':<46}| "
#                 f"{'AAE + RF':<{COL}}| "
#                 f"{'BiGAN':<{COL}}| "
#                 f"{'Robust VAE-BiGAN':<{COL}}| "
#                 f"OC-SVM")

#     def divider():
#         print("  " + "-" * (W - 2))

#     print("=" * W)
#     print(f"{'COMPARISON - AAE vs BiGAN vs Robust VAE-BiGAN vs OC-SVM':^{W}}")
#     print("=" * W)
#     print(f"  DR = Detection Rate   FAR = False Alarm Rate   "
#           f"TP/TN/FP/FN = sample counts\n")
#     print(f"  OC-SVM is trained on benign-only data (semi-supervised), "
#           f"following Amara Korba et al. (2025).\n")

#     # -----------------------------------------------------------------------
#     # [1] BASELINE — Real Test Split
#     # -----------------------------------------------------------------------
#     print(f"  [1] BASELINE - Real Test Split")
#     print(f"  {n_benign_db} benign vs {n_attack_db} attacks "
#           f"({n_attack_db / len(X_db) * 100:.1f}% attacks).\n")

#     p_aae,   _ = predict_aae(aae_data,    X_db)
#     p_bg,    _ = predict_bigan(bigan_data,  X_db)
#     p_rob,   _ = predict_robust(robust_data, X_db)
#     p_ocsvm, _ = predict_ocsvm(ocsvm_data,  X_db)

#     print(f"  {'Model':<26} Result")
#     divider()
#     print(f"  {'AAE + RF':<26} {fmt_honest(y_db_bin, p_aae)}")
#     print(f"  {'BiGAN':<26} {fmt_honest(y_db_bin, p_bg)}")
#     print(f"  {'Robust VAE-BiGAN':<26} {fmt_honest(y_db_bin, p_rob)}")
#     print(f"  {'One-Class SVM':<26} {fmt_honest(y_db_bin, p_ocsvm)}")
#     print()
#     print(f"  OC-SVM note: Trained on benign traffic only.  "
#           f"Detects any deviation from normal IoT behaviour,")
#     print(f"  including UNKNOWN botnets not seen during training "
#           f"(semi-supervised paradigm).")
#     print(f"  AAE note: C&C-HeartBeat has low recall in some runs - only 11 test")
#     print(f"  samples; classifier cannot generalise from that alone.")
#     print("-" * W)

#     # -----------------------------------------------------------------------
#     # [2] STREAM MODE — 70% benign / 30% attacks
#     # -----------------------------------------------------------------------
#     print(f"\n  [2] STREAM MODE - 70% benign / 30% attacks")
#     print(header_row())
#     divider()

#     for n in [200, 500]:
#         X_s, y_s, _ = gateway.generate(mode="stream", n_samples=n)
#         y_s_bin      = (y_s > 0).astype(int)

#         p_aae,   _ = predict_aae(aae_data,    X_s)
#         p_bg,    _ = predict_bigan(bigan_data,  X_s)
#         p_rob,   _ = predict_robust(robust_data, X_s)
#         p_ocsvm, _ = predict_ocsvm(ocsvm_data,  X_s)

#         lbl = f"  {'Stream (n='+str(n)+')':<44}"
#         print(f"{lbl}| {fmt(y_s_bin, p_aae):<{COL}}| "
#               f"{fmt(y_s_bin, p_bg):<{COL}}| "
#               f"{fmt(y_s_bin, p_rob):<{COL}}| "
#               f"{fmt(y_s_bin, p_ocsvm)}")

#     print("-" * W)

#     # -----------------------------------------------------------------------
#     # [3] ADVERSARIAL EVASION — White-box FGSM ε=0.15
#     # -----------------------------------------------------------------------
#     print(f"\n  [3] ADVERSARIAL EVASION - White-box FGSM ε=0.15")
#     print(f"  Note: OC-SVM uses BiGAN-targeted FGSM perturbations "
#           f"(sklearn is not differentiable).")
#     print(header_row())
#     divider()

#     stress_attacks = ["DDoS", "Okiru", "PartOfAHorizontalPortScan"]

#     for atk in stress_attacks:
#         # Generate adversarial samples targeted at each model
#         X_adv_bg,  y_adv, _ = gateway.generate(
#             mode="adversarial", n_samples=300, attack_type=atk,
#             ids_model=bigan_model,   model_type="bigan")

#         X_adv_aae, _, _ = gateway.generate(
#             mode="adversarial", n_samples=300, attack_type=atk,
#             ids_model=aae_model,     model_type="aae")

#         X_adv_rob, _, _ = gateway.generate(
#             mode="adversarial", n_samples=300, attack_type=atk,
#             ids_model=robust_model,  model_type="robust")

#         # OC-SVM: use BiGAN-targeted adversarial samples (black-box proxy)
#         # This is the standard black-box transfer attack evaluation.
#         X_adv_ocsvm = X_adv_bg

#         y_adv_bin = (y_adv > 0).astype(int)

#         p_aae,   _ = predict_aae(aae_data,    X_adv_aae)
#         p_bg,    _ = predict_bigan(bigan_data,  X_adv_bg)
#         p_rob,   _ = predict_robust(robust_data, X_adv_rob)
#         p_ocsvm, _ = predict_ocsvm(ocsvm_data,  X_adv_ocsvm)

#         lbl = f"  {'FGSM ε=0.15 ('+atk+')':<44}"
#         print(f"{lbl}| {fmt(y_adv_bin, p_aae):<{COL}}| "
#               f"{fmt(y_adv_bin, p_bg):<{COL}}| "
#               f"{fmt(y_adv_bin, p_rob):<{COL}}| "
#               f"{fmt(y_adv_bin, p_ocsvm)}")

#     print("-" * W)

#     # -----------------------------------------------------------------------
#     # [4] OC-SVM DETAILED METRICS (per-class breakdown)
#     # -----------------------------------------------------------------------
#     print(f"\n  [4] OC-SVM DETAILED METRICS - Real Test Split")
#     divider()

#     try:
#         with open("saved_state/reverse_label_map.pkl", "rb") as f:
#             rev_map = pickle.load(f)
#         # Invert: code -> name
#         code_to_name = {v: k for k, v in rev_map.items()}
#     except FileNotFoundError:
#         code_to_name = {i: f"class_{i}" for i in range(10)}

#     _, ocsvm_scores = predict_ocsvm(ocsvm_data, X_db)

#     print(f"  {'Class':<28} {'N':>7} {'Detected':>10} {'DR%':>8}")
#     divider()

#     for code in sorted(np.unique(y_db)):
#         mask      = y_db == code
#         n_cls     = int(mask.sum())
#         cls_name  = code_to_name.get(int(code), f"class_{code}")

#         if code == 0:
#             # Benign: measure True Negative Rate (correctly NOT flagged)
#             tn_cls = int((p_ocsvm[mask] == 0).sum())
#             rate   = tn_cls / max(n_cls, 1) * 100
#             print(f"  {'[Benign] '+cls_name:<28} {n_cls:>7} "
#                   f"{tn_cls:>10}  TNR={rate:6.2f}%")
#         else:
#             tp_cls = int(p_ocsvm[mask].sum())
#             rate   = tp_cls / max(n_cls, 1) * 100
#             print(f"  {cls_name:<28} {n_cls:>7} "
#                   f"{tp_cls:>10}  DR={rate:7.2f}%")

#     print()
#     print(f"  OC-SVM overall: {fmt_honest(y_db_bin, p_ocsvm)}")
#     print()

#     # -----------------------------------------------------------------------
#     # [5] SUMMARY TABLE
#     # -----------------------------------------------------------------------
#     print("=" * W)
#     print(f"{'SUMMARY':^{W}}")
#     print("=" * W)

#     all_scenarios = [
#         ("Real Test Data",   y_db_bin, p_aae, p_bg, p_rob, p_ocsvm),
#     ]

#     print(f"  {'Scenario':<26} {'Model':<22} {'DR%':>8} {'FAR%':>8} {'Acc%':>8}")
#     divider()

#     for label, y_bin, pa, pb, pr, po in all_scenarios:
#         for mname, mp in [("AAE+RF", pa), ("BiGAN", pb),
#                           ("Robust VAE-BiGAN", pr), ("OC-SVM", po)]:
#             dr_  = detection_rate(y_bin, mp)
#             far_ = false_alarm_rate(y_bin, mp)
#             acc_ = accuracy_score(y_bin, mp) * 100
#             print(f"  {label:<26} {mname:<22} {dr_:>8.2f} {far_:>8.2f} {acc_:>8.2f}")
#         divider()

#     print()
#     print("  KEY INSIGHT (from Amara Korba et al. 2025):")
#     print("  OC-SVM achieves 99.99% recall on packet-level C&C detection with")
#     print("  only 1.53% false positive rate.  Unlike BiGAN / Robust / AAE,")
#     print("  OC-SVM requires ONLY benign data for training, enabling it to")
#     print("  detect UNKNOWN botnets not seen during model development.")
#     print("=" * W)





























































































# latest code



# """
# comparison2.py
# --------------
# Compare AAE, BiGAN, Robust VAE-BiGAN, and One-Class SVM (OC-SVM).

# OC-SVM is added following:
#   "Mitigating IoT botnet attacks: An early-stage explainable network-based
#    anomaly detection approach" (Amara Korba et al., Computer Communications 2025).

# That paper reports OC-SVM as the top-performing anomaly detector, achieving
# 99.99% recall and 1.53% FPR on packet-level C&C traffic.

# Run after training all models:
#     python train_aae.py
#     python train_aae_classifier.py
#     python train_bigan.py
#     python train_robust.py   (then calibrate_robust.py)
#     python train_ocsvm.py    ← new
#     python comparison2.py
# """

# import torch
# import numpy as np
# import pickle

# from sklearn.metrics import accuracy_score

# from models.bigan_model  import BiGAN
# from models.robust_model import RobustVAEBiGAN
# from models.aae_model    import AAE
# from models.ocsvm_model  import OCSVMDetector   # ← new
# from gateway.gateway     import Gateway


# # ---------------------------------------------------------------------------
# # Model loading helpers
# # ---------------------------------------------------------------------------

# def load_all_models(input_dim):
#     """Load BiGAN, Robust VAE-BiGAN, AAE, and OC-SVM."""

#     # -- BiGAN ---------------------------------------------------------------
#     bigan   = BiGAN(input_dim)
#     bg_ckpt = torch.load("bigan_final.pth", map_location="cpu")
#     bigan.load_state_dict(
#         bg_ckpt["state_dict"] if "state_dict" in bg_ckpt else bg_ckpt)
#     bigan.eval()

#     with open("bigan_calibration.pkl", "rb") as f:
#         bg_calib = pickle.load(f)

#     # -- Robust VAE-BiGAN ----------------------------------------------------
#     robust = RobustVAEBiGAN(input_dim)
#     robust.load_state_dict(
#         torch.load("saved_state/robust_vae_bigan_model.pth", map_location="cpu"))
#     robust.eval()

#     try:
#         with open("robust_calibration.pkl", "rb") as f:
#             rob_calib = pickle.load(f)
#         print(f"  Robust calibration: threshold={rob_calib['threshold']:.4f}  "
#               f"flip={rob_calib['flip']}")
#     except FileNotFoundError:
#         raise FileNotFoundError(
#             "robust_calibration.pkl not found. "
#             "Run calibrate_robust.py (or train_robust.py) first.")

#     # -- AAE + RF ------------------------------------------------------------
#     aae      = AAE(input_dim=input_dim)
#     aae_ckpt = torch.load("aae_final.pth", map_location="cpu")
#     aae.load_state_dict(aae_ckpt["state_dict"])
#     aae.eval()

#     with open("aae_classifier.pkl", "rb") as f:
#         aae_rf = pickle.load(f)

#     # -- One-Class SVM -------------------------------------------------------
#     try:
#         ocsvm    = OCSVMDetector.load("ocsvm_model.pkl")
#         with open("ocsvm_calibration.pkl", "rb") as f:
#             ocsvm_calib = pickle.load(f)
#         print(f"  OC-SVM calibration: threshold={ocsvm_calib['threshold']:.4f}  "
#               f"nu={ocsvm_calib['nu']}  gamma={ocsvm_calib['gamma']}")
#     except FileNotFoundError:
#         raise FileNotFoundError(
#             "ocsvm_model.pkl or ocsvm_calibration.pkl not found. "
#             "Run train_ocsvm.py first.")

#     return (
#         (bigan,  bg_calib),
#         (robust, rob_calib),
#         (aae,    aae_rf),
#         (ocsvm,  ocsvm_calib),
#     )


# # ---------------------------------------------------------------------------
# # Prediction helpers
# # ---------------------------------------------------------------------------

# def predict_bigan(model_data, X):
#     model, calib = model_data
#     X_t = torch.FloatTensor(X.astype(np.float32))
#     with torch.no_grad():
#         z       = model.encoder(X_t)
#         recon   = torch.mean((X_t - model.generator(z)) ** 2, dim=1).numpy()
#         d_score = model.discriminator(
#             torch.cat([X_t, z], dim=1)).squeeze().numpy()
#         if d_score.ndim == 0:
#             d_score = d_score.reshape(1)

#     span       = calib["r_max"] - calib["r_min"]
#     recon_norm = np.clip((recon - calib["r_min"]) / (span + 1e-8), 0.0, 1.0)
#     raw        = 0.5 * (1.0 - d_score) + 0.5 * recon_norm
#     scores     = (1.0 - raw) if calib["flip"] else raw
#     return (scores > calib["threshold"]).astype(int), scores


# def predict_robust(model_data, X):
#     model, calib = model_data
#     X_t   = torch.FloatTensor(X.astype(np.float32))
#     all_s = []
#     with torch.no_grad():
#         for i in range(0, len(X_t), 1000):
#             batch = X_t[i:i+1000]
#             mu, _ = model.encode(batch)
#             s     = model.discriminator(
#                 torch.cat([batch, mu], dim=1)).squeeze().numpy()
#             if s.ndim == 0:
#                 s = s.reshape(1)
#             all_s.append(s)
#     scores = np.concatenate(all_s)
#     if calib["flip"]:
#         scores = 1.0 - scores
#     return (scores > calib["threshold"]).astype(int), scores


# def predict_aae(model_data, X):
#     aae, rf = model_data
#     X_t = torch.FloatTensor(X.astype(np.float32))
#     with torch.no_grad():
#         Z = aae.encoder(X_t).numpy()
#     mc_preds = rf.predict(Z)
#     probas   = rf.predict_proba(Z).max(axis=1)
#     return (mc_preds > 0).astype(int), probas


# def predict_ocsvm(model_data, X):
#     """
#     One-Class SVM prediction.

#     model_data = (OCSVMDetector, calib_dict)
#     Returns (preds, anomaly_scores) with preds in {0, 1}.
#     """
#     detector, _calib = model_data
#     preds, scores = detector.predict(X.astype(np.float32))
#     return preds, scores


# # ---------------------------------------------------------------------------
# # Metric helpers
# # ---------------------------------------------------------------------------

# def detection_rate(y_true_bin, y_pred):
#     attacks = y_true_bin == 1
#     if attacks.sum() == 0:
#         return 0.0
#     return float(y_pred[attacks].mean()) * 100


# def false_alarm_rate(y_true_bin, y_pred):
#     benign = y_true_bin == 0
#     if benign.sum() == 0:
#         return 0.0
#     return float(y_pred[benign].mean()) * 100


# def fmt(y_true_bin, y_pred):
#     """One-liner: Acc / DR / FAR."""
#     acc = accuracy_score(y_true_bin, y_pred) * 100
#     dr  = detection_rate(y_true_bin, y_pred)
#     far = false_alarm_rate(y_true_bin, y_pred)
#     return f"Acc={acc:5.1f}%  DR={dr:5.1f}%  FAR={far:4.1f}%"


# def fmt_honest(y_true_bin, y_pred):
#     """Show TP/TN/FP/FN counts."""
#     attacks = y_true_bin == 1
#     benign  = y_true_bin == 0
#     tp = int(y_pred[attacks].sum())
#     tn = int((y_pred[benign] == 0).sum())
#     fp = int(y_pred[benign].sum())
#     fn = int((y_pred[attacks] == 0).sum())
#     dr  = tp / max(attacks.sum(), 1) * 100
#     far = fp / max(benign.sum(),  1) * 100
#     return f"DR={dr:5.1f}%  FAR={far:4.1f}%  TP={tp}  TN={tn}  FP={fp}  FN={fn}"


# # ---------------------------------------------------------------------------
# # Main comparison
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":

#     print("Loading test data and models...")

#     test_data  = np.load("test_data.npz")
#     X_db, y_db = test_data["X_test"], test_data["y_test"]
#     y_db_bin   = (y_db > 0).astype(int)
#     input_dim  = X_db.shape[1]

#     n_benign_db = int((y_db_bin == 0).sum())
#     n_attack_db = int((y_db_bin == 1).sum())

#     print(f"  input_dim={input_dim}  test_samples={len(X_db)}")
#     print(f"  Benign={n_benign_db}  Attack={n_attack_db}  "
#           f"(ratio 1:{n_attack_db // max(n_benign_db, 1)})\n")

#     bigan_data, robust_data, aae_data, ocsvm_data = load_all_models(input_dim)

#     bigan_model  = bigan_data[0]
#     robust_model = robust_data[0]
#     aae_model    = aae_data[0]
#     # OC-SVM does not need a separate model object for Gateway FGSM
#     # (sklearn models are not differentiable), so adversarial mode
#     # for OC-SVM uses BiGAN's FGSM perturbations for a fair comparison.

#     print("  All models loaded.\n")

#     gateway = Gateway(epsilon=0.15, gan_blend_alpha=0.0)

#     W   = 160          # total line width
#     COL = 38           # column width per model result

#     def header_row():
#         return (f"  {'SCENARIO':<46}| "
#                 f"{'AAE + RF':<{COL}}| "
#                 f"{'BiGAN':<{COL}}| "
#                 f"{'Robust VAE-BiGAN':<{COL}}| "
#                 f"OC-SVM")

#     def divider():
#         print("  " + "-" * (W - 2))

#     print("=" * W)
#     print(f"{'COMPARISON - AAE vs BiGAN vs Robust VAE-BiGAN vs OC-SVM':^{W}}")
#     print("=" * W)
#     print(f"  DR = Detection Rate   FAR = False Alarm Rate   "
#           f"TP/TN/FP/FN = sample counts\n")
#     print(f"  OC-SVM is trained on benign-only data (semi-supervised), "
#           f"following Amara Korba et al. (2025).\n")

#     # -----------------------------------------------------------------------
#     # [1] BASELINE — Real Test Split
#     # -----------------------------------------------------------------------
#     print(f"  [1] BASELINE - Real Test Split")
#     print(f"  {n_benign_db} benign vs {n_attack_db} attacks "
#           f"({n_attack_db / len(X_db) * 100:.1f}% attacks).\n")

#     # Store with _db suffix — these must NEVER be overwritten by loop variables
#     p_aae_db,   _ = predict_aae(aae_data,     X_db)
#     p_bg_db,    _ = predict_bigan(bigan_data,  X_db)
#     p_rob_db,   _ = predict_robust(robust_data, X_db)
#     p_ocsvm_db, _ = predict_ocsvm(ocsvm_data,  X_db)

#     print(f"  {'Model':<26} Result")
#     divider()
#     print(f"  {'AAE + RF':<26} {fmt_honest(y_db_bin, p_aae_db)}")
#     print(f"  {'BiGAN':<26} {fmt_honest(y_db_bin, p_bg_db)}")
#     print(f"  {'Robust VAE-BiGAN':<26} {fmt_honest(y_db_bin, p_rob_db)}")
#     print(f"  {'One-Class SVM':<26} {fmt_honest(y_db_bin, p_ocsvm_db)}")
#     print()
#     print(f"  OC-SVM note: Trained on benign traffic only.  "
#           f"Detects any deviation from normal IoT behaviour,")
#     print(f"  including UNKNOWN botnets not seen during training "
#           f"(semi-supervised paradigm).")
#     print(f"  AAE note: C&C-HeartBeat has low recall in some runs - only 11 test")
#     print(f"  samples; classifier cannot generalise from that alone.")
#     print("-" * W)

#     # -----------------------------------------------------------------------
#     # [2] STREAM MODE — 70% benign / 30% attacks
#     # -----------------------------------------------------------------------
#     print(f"\n  [2] STREAM MODE - 70% benign / 30% attacks")
#     print(header_row())
#     divider()

#     for n in [200, 500]:
#         X_s, y_s, _ = gateway.generate(mode="stream", n_samples=n)
#         y_s_bin      = (y_s > 0).astype(int)

#         p_aae,   _ = predict_aae(aae_data,    X_s)
#         p_bg,    _ = predict_bigan(bigan_data,  X_s)
#         p_rob,   _ = predict_robust(robust_data, X_s)
#         p_ocsvm, _ = predict_ocsvm(ocsvm_data,  X_s)

#         lbl = f"  {'Stream (n='+str(n)+')':<44}"
#         print(f"{lbl}| {fmt(y_s_bin, p_aae):<{COL}}| "
#               f"{fmt(y_s_bin, p_bg):<{COL}}| "
#               f"{fmt(y_s_bin, p_rob):<{COL}}| "
#               f"{fmt(y_s_bin, p_ocsvm)}")

#     print("-" * W)

#     # -----------------------------------------------------------------------
#     # [3] ADVERSARIAL EVASION — White-box FGSM ε=0.15
#     # -----------------------------------------------------------------------
#     print(f"\n  [3] ADVERSARIAL EVASION - White-box FGSM ε=0.15")
#     print(f"  Note: OC-SVM uses BiGAN-targeted FGSM perturbations "
#           f"(sklearn is not differentiable).")
#     print(header_row())
#     divider()

#     stress_attacks = ["DDoS", "Okiru", "PartOfAHorizontalPortScan"]

#     for atk in stress_attacks:
#         # Generate adversarial samples targeted at each model
#         X_adv_bg,  y_adv, _ = gateway.generate(
#             mode="adversarial", n_samples=300, attack_type=atk,
#             ids_model=bigan_model,   model_type="bigan")

#         X_adv_aae, _, _ = gateway.generate(
#             mode="adversarial", n_samples=300, attack_type=atk,
#             ids_model=aae_model,     model_type="aae")

#         X_adv_rob, _, _ = gateway.generate(
#             mode="adversarial", n_samples=300, attack_type=atk,
#             ids_model=robust_model,  model_type="robust")

#         # OC-SVM: use BiGAN-targeted adversarial samples (black-box proxy)
#         # This is the standard black-box transfer attack evaluation.
#         X_adv_ocsvm = X_adv_bg

#         y_adv_bin = (y_adv > 0).astype(int)

#         p_aae,   _ = predict_aae(aae_data,    X_adv_aae)
#         p_bg,    _ = predict_bigan(bigan_data,  X_adv_bg)
#         p_rob,   _ = predict_robust(robust_data, X_adv_rob)
#         p_ocsvm, _ = predict_ocsvm(ocsvm_data,  X_adv_ocsvm)

#         lbl = f"  {'FGSM ε=0.15 ('+atk+')':<44}"
#         print(f"{lbl}| {fmt(y_adv_bin, p_aae):<{COL}}| "
#               f"{fmt(y_adv_bin, p_bg):<{COL}}| "
#               f"{fmt(y_adv_bin, p_rob):<{COL}}| "
#               f"{fmt(y_adv_bin, p_ocsvm)}")

#     print("-" * W)

#     # -----------------------------------------------------------------------
#     # [4] OC-SVM DETAILED METRICS (per-class breakdown)
#     # -----------------------------------------------------------------------
#     print(f"\n  [4] OC-SVM DETAILED METRICS - Real Test Split")
#     divider()

#     try:
#         with open("saved_state/reverse_label_map.pkl", "rb") as f:
#             rev_map = pickle.load(f)
#         # Invert: code -> name
#         code_to_name = {v: k for k, v in rev_map.items()}
#     except FileNotFoundError:
#         code_to_name = {i: f"class_{i}" for i in range(10)}

#     # Recompute on full X_db — p_ocsvm was overwritten in the adversarial
#     # loop above (300 samples each), so we need a fresh prediction here.
#     p_ocsvm_db, ocsvm_scores = predict_ocsvm(ocsvm_data, X_db)

#     print(f"  {'Class':<28} {'N':>7} {'Detected':>10} {'DR%':>8}")
#     divider()

#     for code in sorted(np.unique(y_db)):
#         mask      = y_db == code
#         n_cls     = int(mask.sum())
#         cls_name  = code_to_name.get(int(code), f"class_{code}")

#         if code == 0:
#             # Benign: measure True Negative Rate (correctly NOT flagged)
#             tn_cls = int((p_ocsvm_db[mask] == 0).sum())
#             rate   = tn_cls / max(n_cls, 1) * 100
#             print(f"  {'[Benign] '+cls_name:<28} {n_cls:>7} "
#                   f"{tn_cls:>10}  TNR={rate:6.2f}%")
#         else:
#             tp_cls = int(p_ocsvm_db[mask].sum())
#             rate   = tp_cls / max(n_cls, 1) * 100
#             print(f"  {cls_name:<28} {n_cls:>7} "
#                   f"{tp_cls:>10}  DR={rate:7.2f}%")

#     print()
#     print(f"  OC-SVM overall: {fmt_honest(y_db_bin, p_ocsvm_db)}")
#     print()

#     # -----------------------------------------------------------------------
#     # [5] SUMMARY TABLE
#     # -----------------------------------------------------------------------
#     print("=" * W)
#     print(f"{'SUMMARY':^{W}}")
#     print("=" * W)

#     all_scenarios = [
#         ("Real Test Data",   y_db_bin, p_aae_db, p_bg_db, p_rob_db, p_ocsvm_db),
#     ]

#     print(f"  {'Scenario':<26} {'Model':<22} {'DR%':>8} {'FAR%':>8} {'Acc%':>8}")
#     divider()

#     for label, y_bin, pa, pb, pr, po in all_scenarios:
#         for mname, mp in [("AAE+RF", pa), ("BiGAN", pb),
#                           ("Robust VAE-BiGAN", pr), ("OC-SVM", po)]:
#             dr_  = detection_rate(y_bin, mp)
#             far_ = false_alarm_rate(y_bin, mp)
#             acc_ = accuracy_score(y_bin, mp) * 100
#             print(f"  {label:<26} {mname:<22} {dr_:>8.2f} {far_:>8.2f} {acc_:>8.2f}")
#         divider()

#     print()
#     print("  KEY INSIGHT (from Amara Korba et al. 2025):")
#     print("  OC-SVM achieves 99.99% recall on packet-level C&C detection with")
#     print("  only 1.53% false positive rate.  Unlike BiGAN / Robust / AAE,")
#     print("  OC-SVM requires ONLY benign data for training, enabling it to")
#     print("  detect UNKNOWN botnets not seen during model development.")
#     print("=" * W)



















































































"""
comparison2.py
--------------
Compare AAE, BiGAN, Robust VAE-BiGAN, and One-Class SVM (OC-SVM).
"""

import logging
import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from models.bigan_model  import BiGAN
from models.robust_model import RobustVAEBiGAN
from models.aae_model    import AAE
from models.ocsvm_model  import OCSVMDetector
from gateway.gateway     import Gateway

# Suppress all INFO logs from gateway / model loaders
logging.disable(logging.CRITICAL)


def load_all_models(input_dim):
    bigan   = BiGAN(input_dim)
    bg_ckpt = torch.load("bigan_final.pth", map_location="cpu")
    bigan.load_state_dict(bg_ckpt["state_dict"] if "state_dict" in bg_ckpt else bg_ckpt)
    bigan.eval()
    with open("bigan_calibration.pkl", "rb") as f:
        bg_calib = pickle.load(f)

    robust = RobustVAEBiGAN(input_dim)
    robust.load_state_dict(torch.load("saved_state/robust_vae_bigan_model.pth", map_location="cpu"))
    robust.eval()
    with open("robust_calibration.pkl", "rb") as f:
        rob_calib = pickle.load(f)

    aae      = AAE(input_dim=input_dim)
    aae_ckpt = torch.load("aae_final.pth", map_location="cpu")
    aae.load_state_dict(aae_ckpt["state_dict"])
    aae.eval()
    with open("aae_classifier.pkl", "rb") as f:
        aae_rf = pickle.load(f)

    ocsvm = OCSVMDetector.load("ocsvm_model.pkl")
    with open("ocsvm_calibration.pkl", "rb") as f:
        ocsvm_calib = pickle.load(f)

    return (bigan, bg_calib), (robust, rob_calib), (aae, aae_rf), (ocsvm, ocsvm_calib)


def predict_bigan(model_data, X):
    model, calib = model_data
    X_t = torch.FloatTensor(X.astype(np.float32))
    with torch.no_grad():
        z       = model.encoder(X_t)
        recon   = torch.mean((X_t - model.generator(z)) ** 2, dim=1).numpy()
        d_score = model.discriminator(torch.cat([X_t, z], dim=1)).squeeze().numpy()
        if d_score.ndim == 0:
            d_score = d_score.reshape(1)
    span       = calib["r_max"] - calib["r_min"]
    recon_norm = np.clip((recon - calib["r_min"]) / (span + 1e-8), 0.0, 1.0)
    raw        = 0.5 * (1.0 - d_score) + 0.5 * recon_norm
    scores     = (1.0 - raw) if calib["flip"] else raw
    return (scores > calib["threshold"]).astype(int), scores


def predict_robust(model_data, X):
    model, calib = model_data
    X_t = torch.FloatTensor(X.astype(np.float32))
    all_s = []
    with torch.no_grad():
        for i in range(0, len(X_t), 1000):
            batch = X_t[i:i+1000]
            mu, _ = model.encode(batch)
            s     = model.discriminator(torch.cat([batch, mu], dim=1)).squeeze().numpy()
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


def predict_ocsvm(model_data, X):
    detector, _ = model_data
    return detector.predict(X.astype(np.float32))


def detection_rate(y_bin, y_pred):
    attacks = y_bin == 1
    return float(y_pred[attacks].mean()) * 100 if attacks.sum() > 0 else 0.0


def false_alarm_rate(y_bin, y_pred):
    benign = y_bin == 0
    return float(y_pred[benign].mean()) * 100 if benign.sum() > 0 else 0.0


def fmt(y_bin, y_pred):
    acc = accuracy_score(y_bin, y_pred) * 100
    dr  = detection_rate(y_bin, y_pred)
    far = false_alarm_rate(y_bin, y_pred)
    return f"Acc={acc:5.1f}%  DR={dr:5.1f}%  FAR={far:4.1f}%"


def fmt_honest(y_bin, y_pred):
    attacks = y_bin == 1
    benign  = y_bin == 0
    tp = int(y_pred[attacks].sum())
    tn = int((y_pred[benign] == 0).sum())
    fp = int(y_pred[benign].sum())
    fn = int((y_pred[attacks] == 0).sum())
    dr  = tp / max(attacks.sum(), 1) * 100
    far = fp / max(benign.sum(),  1) * 100
    return f"DR={dr:5.1f}%  FAR={far:4.1f}%  TP={tp}  TN={tn}  FP={fp}  FN={fn}"


if __name__ == "__main__":

    test_data  = np.load("test_data.npz")
    X_db, y_db = test_data["X_test"], test_data["y_test"]
    y_db_bin   = (y_db > 0).astype(int)
    input_dim  = X_db.shape[1]

    n_benign_db = int((y_db_bin == 0).sum())
    n_attack_db = int((y_db_bin == 1).sum())

    bigan_data, robust_data, aae_data, ocsvm_data = load_all_models(input_dim)
    bigan_model  = bigan_data[0]
    robust_model = robust_data[0]
    aae_model    = aae_data[0]

    gateway = Gateway(epsilon=0.15, gan_blend_alpha=0.0)

    W   = 158
    COL = 38

    def divider():
        print("  " + "-" * (W - 2))

    def header_row():
        return (f"  {'SCENARIO':<46}| "
                f"{'AAE + RF':<{COL}}| "
                f"{'BiGAN':<{COL}}| "
                f"{'Robust VAE-BiGAN':<{COL}}| "
                f"OC-SVM")

    print("=" * W)
    print(f"{'COMPARISON - AAE vs BiGAN vs Robust VAE-BiGAN vs OC-SVM':^{W}}")
    print("=" * W)
    print(f"  DR = Detection Rate   FAR = False Alarm Rate   TP/TN/FP/FN = sample counts")
    print()

    # [1] BASELINE
    print(f"  [1] BASELINE - Real Test Split")
    print(f"  {n_benign_db} benign  |  {n_attack_db} attacks  |  {len(X_db)} total samples")
    print()

    p_aae_db,   _ = predict_aae(aae_data,      X_db)
    p_bg_db,    _ = predict_bigan(bigan_data,   X_db)
    p_rob_db,   _ = predict_robust(robust_data, X_db)
    p_ocsvm_db, _ = predict_ocsvm(ocsvm_data,   X_db)

    print(f"  {'Model':<26} Result")
    divider()
    print(f"  {'AAE + RF':<26} {fmt_honest(y_db_bin, p_aae_db)}")
    print(f"  {'BiGAN':<26} {fmt_honest(y_db_bin, p_bg_db)}")
    print(f"  {'Robust VAE-BiGAN':<26} {fmt_honest(y_db_bin, p_rob_db)}")
    print(f"  {'One-Class SVM':<26} {fmt_honest(y_db_bin, p_ocsvm_db)}")
    print()

    # [2] STREAM MODE
    print(f"  [2] STREAM MODE - 70% benign / 30% attacks")
    print(header_row())
    divider()

    for n in [200, 500]:
        X_s, y_s, _ = gateway.generate(mode="stream", n_samples=n)
        y_s_bin = (y_s > 0).astype(int)
        p_aae,   _ = predict_aae(aae_data,      X_s)
        p_bg,    _ = predict_bigan(bigan_data,   X_s)
        p_rob,   _ = predict_robust(robust_data, X_s)
        p_ocsvm, _ = predict_ocsvm(ocsvm_data,  X_s)
        lbl = f"  {'Stream (n='+str(n)+')':<44}"
        print(f"{lbl}| {fmt(y_s_bin, p_aae):<{COL}}| "
              f"{fmt(y_s_bin, p_bg):<{COL}}| "
              f"{fmt(y_s_bin, p_rob):<{COL}}| "
              f"{fmt(y_s_bin, p_ocsvm)}")

    print()

    # [3] ADVERSARIAL EVASION
    print(f"  [3] ADVERSARIAL EVASION - White-box FGSM epsilon=0.15")
    print(f"  OC-SVM column: BiGAN-targeted perturbations (black-box transfer attack).")
    print(header_row())
    divider()

    for atk in ["DDoS", "Okiru", "PartOfAHorizontalPortScan"]:
        X_adv_bg,  y_adv, _ = gateway.generate(mode="adversarial", n_samples=300,
                                                attack_type=atk, ids_model=bigan_model,
                                                model_type="bigan")
        X_adv_aae, _, _     = gateway.generate(mode="adversarial", n_samples=300,
                                                attack_type=atk, ids_model=aae_model,
                                                model_type="aae")
        X_adv_rob, _, _     = gateway.generate(mode="adversarial", n_samples=300,
                                                attack_type=atk, ids_model=robust_model,
                                                model_type="robust")
        y_adv_bin = (y_adv > 0).astype(int)
        p_aae,   _ = predict_aae(aae_data,      X_adv_aae)
        p_bg,    _ = predict_bigan(bigan_data,   X_adv_bg)
        p_rob,   _ = predict_robust(robust_data, X_adv_rob)
        p_ocsvm, _ = predict_ocsvm(ocsvm_data,   X_adv_bg)
        lbl = f"  {'FGSM e=0.15 ('+atk+')':<44}"
        print(f"{lbl}| {fmt(y_adv_bin, p_aae):<{COL}}| "
              f"{fmt(y_adv_bin, p_bg):<{COL}}| "
              f"{fmt(y_adv_bin, p_rob):<{COL}}| "
              f"{fmt(y_adv_bin, p_ocsvm)}")

    print()

    print("=" * W)
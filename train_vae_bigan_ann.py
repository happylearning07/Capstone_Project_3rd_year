# """
# train_vae_bigan_ann.py
# ======================
# Joint training of VAE-BiGAN + ANN classifier for multiclass classification.

# The key idea:
#   - A fresh VAE-BiGAN encoder is trained from scratch (recon + KL + GAN losses)
#   - SIMULTANEOUSLY, an ANN head on top of z = mu is trained with CrossEntropy
#   - Classification gradients flow back through the encoder, so z becomes
#     both a good generative latent AND class-discriminative

# Compatible with YOUR existing project:
#   - Reads from  : test_data.npz                      (already in your folder)
#   - Reads from  : saved_state/reverse_label_map.pkl  (already in your folder)
#   - y_train / y_test are INTEGER codes — handled correctly here
#   - Saves to    : saved_state/vae_bigan_ann_model.pth
#                   saved_state/ann_classifier.pth
#                   saved_state/ann_multiclass_meta.pkl

# Run:
#     python train_vae_bigan_ann.py
# """

# import os
# import pickle
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split

# # ─────────────────────────────────────────────────────────────────
# # Paths  (relative to final_capstone/)
# # ─────────────────────────────────────────────────────────────────
# DATA_PATH         = "test_data.npz"
# REVERSE_LABEL_MAP = "saved_state/reverse_label_map.pkl"

# SAVE_MODEL_PATH   = "saved_state/vae_bigan_ann_model.pth"
# SAVE_CLF_PATH     = "saved_state/ann_classifier.pth"
# SAVE_META_PATH    = "saved_state/ann_multiclass_meta.pkl"


# # ─────────────────────────────────────────────────────────────────
# # 1. Architecture
# # ─────────────────────────────────────────────────────────────────

# class Encoder(nn.Module):
#     def __init__(self, input_dim: int, latent_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
#             nn.Linear(128, 64),        nn.LeakyReLU(0.2), nn.BatchNorm1d(64),
#         )
#         self.fc_mu     = nn.Linear(64, latent_dim)
#         self.fc_logvar = nn.Linear(64, latent_dim)

#     def forward(self, x):
#         h = self.net(x)
#         return self.fc_mu(h), self.fc_logvar(h)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             return mu + torch.randn_like(std) * std
#         return mu


# class Decoder(nn.Module):
#     def __init__(self, latent_dim: int, output_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, 64),  nn.LeakyReLU(0.2), nn.BatchNorm1d(64),
#             nn.Linear(64, 128),         nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
#             nn.Linear(128, output_dim), nn.Sigmoid(),
#         )

#     def forward(self, z):
#         return self.net(z)


# class Discriminator(nn.Module):
#     def __init__(self, input_dim: int, latent_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim + latent_dim, 64), nn.LeakyReLU(0.2), nn.Dropout(0.3),
#             nn.Linear(64, 32),                      nn.LeakyReLU(0.2), nn.Dropout(0.3),
#             nn.Linear(32, 1),                       nn.Sigmoid(),
#         )

#     def forward(self, x, z):
#         return self.net(torch.cat([x, z], dim=1))


# class ANNClassifier(nn.Module):
#     """Multiclass head on top of latent z."""
#     def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
#             nn.Linear(64, 32),         nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout),
#             nn.Linear(32, num_classes),
#         )

#     def forward(self, z):
#         return self.net(z)


# # ─────────────────────────────────────────────────────────────────
# # 2. Loss helper
# # ─────────────────────────────────────────────────────────────────

# def vae_loss(x, x_recon, mu, logvar, beta=1.0):
#     recon = nn.functional.mse_loss(x_recon, x, reduction="mean")
#     kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon + beta * kl


# # ─────────────────────────────────────────────────────────────────
# # 3. Data loading
# # ─────────────────────────────────────────────────────────────────

# def load_data():
#     if not os.path.exists(DATA_PATH):
#         raise FileNotFoundError(
#             f"'{DATA_PATH}' not found.\n"
#             "Make sure you are running this script from inside final_capstone/."
#         )
#     if not os.path.exists(REVERSE_LABEL_MAP):
#         raise FileNotFoundError(
#             f"'{REVERSE_LABEL_MAP}' not found.\n"
#             "Run train_aae.py or train_bigan.py first to generate it."
#         )

#     data    = np.load(DATA_PATH)
#     X_train = data["X_train"].astype(np.float32)
#     y_train = data["y_train"].astype(np.int64)
#     X_test  = data["X_test"].astype(np.float32)
#     y_test  = data["y_test"].astype(np.int64)

#     with open(REVERSE_LABEL_MAP, "rb") as f:
#         name_to_code = pickle.load(f)           # {name: int_code}
#     code_to_name = {v: k for k, v in name_to_code.items()}

#     # Remap to consecutive 0..N-1 in case codes have gaps
#     unique_codes = np.unique(y_train)
#     code_to_idx  = {int(c): i for i, c in enumerate(unique_codes)}
#     idx_to_name  = {i: code_to_name.get(int(c), str(c))
#                     for i, c in enumerate(unique_codes)}

#     y_train_idx = np.array([code_to_idx[int(c)] for c in y_train], dtype=np.int64)
#     y_test_idx  = np.array([code_to_idx.get(int(c), -1) for c in y_test], dtype=np.int64)

#     num_classes = len(unique_codes)

#     print(f"  input_dim   : {X_train.shape[1]}")
#     print(f"  num_classes : {num_classes}")
#     print(f"  Classes:")
#     for idx, name in idx_to_name.items():
#         n_tr = int((y_train_idx == idx).sum())
#         n_te = int((y_test_idx  == idx).sum())
#         print(f"    [{idx}] {name:<30}  train={n_tr:>6}  test={n_te:>5}")
#     print(f"  train total : {len(X_train)}   test total : {len(X_test)}\n")

#     return (X_train, y_train_idx,
#             X_test,  y_test_idx,
#             num_classes, idx_to_name, code_to_idx)


# # ─────────────────────────────────────────────────────────────────
# # 4. Training
# # ─────────────────────────────────────────────────────────────────

# def train(args):
#     os.makedirs("saved_state", exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n[INFO] Device     : {device}")
#     print(f"[INFO] Data file  : {DATA_PATH}")
#     print("[INFO] Loading data ...\n")

#     (X_train, y_train,
#      X_test,  y_test,
#      num_classes, idx_to_name, code_to_idx) = load_data()

#     input_dim  = X_train.shape[1]
#     latent_dim = args.latent_dim

#     # Val split from training data
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
#     )

#     tr_dl = DataLoader(
#         TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
#         batch_size=args.batch_size, shuffle=True, drop_last=True,
#     )
#     val_dl = DataLoader(
#         TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
#         batch_size=512, shuffle=False,
#     )

#     # Models
#     encoder       = Encoder(input_dim, latent_dim).to(device)
#     decoder       = Decoder(latent_dim, input_dim).to(device)
#     discriminator = Discriminator(input_dim, latent_dim).to(device)
#     classifier    = ANNClassifier(latent_dim, num_classes, dropout=args.dropout).to(device)

#     opt_gen  = optim.Adam(
#         list(encoder.parameters()) +
#         list(decoder.parameters()) +
#         list(classifier.parameters()),
#         lr=args.lr, betas=(0.5, 0.999),
#     )
#     opt_disc = optim.Adam(
#         discriminator.parameters(),
#         lr=args.lr * 0.5, betas=(0.5, 0.999),
#     )

#     ce_loss  = nn.CrossEntropyLoss()
#     bce_loss = nn.BCELoss()
#     sched    = optim.lr_scheduler.StepLR(opt_gen, step_size=20, gamma=0.5)

#     best_val_acc = 0.0
#     best_enc_state = best_dec_state = best_disc_state = best_clf_state = None

#     print(f"[INFO] Training  epochs={args.epochs}  lr={args.lr}  "
#           f"batch={args.batch_size}  latent_dim={latent_dim}  "
#           f"lambda_cls={args.lambda_cls}\n")
#     print(f"  {'Ep':>4}  {'VAE-Loss':>9}  {'D-Loss':>7}  {'Cls-Loss':>9}  {'Val-Acc':>8}")
#     print("  " + "-" * 50)

#     for epoch in range(1, args.epochs + 1):
#         encoder.train(); decoder.train()
#         discriminator.train(); classifier.train()

#         ep_vae = ep_disc = ep_cls = 0.0

#         for x_b, y_b in tr_dl:
#             x_b = x_b.to(device)
#             y_b = y_b.to(device)
#             B   = x_b.size(0)
#             ones  = torch.ones(B,  1, device=device)
#             zeros = torch.zeros(B, 1, device=device)

#             # Discriminator update (encoder frozen via no_grad)
#             with torch.no_grad():
#                 mu_d, lv_d = encoder(x_b)
#                 z_enc_d    = encoder.reparameterize(mu_d, lv_d)
#                 z_prior    = torch.randn(B, latent_dim, device=device)
#                 x_fake     = decoder(z_prior)

#             opt_disc.zero_grad()
#             d_real = discriminator(x_b,    z_enc_d)
#             d_fake = discriminator(x_fake, z_prior)
#             loss_d = 0.5 * (bce_loss(d_real, ones) + bce_loss(d_fake, zeros))
#             loss_d.backward()
#             opt_disc.step()

#             # Encoder + Decoder + Classifier update
#             opt_gen.zero_grad()

#             mu, logvar = encoder(x_b)
#             z_enc      = encoder.reparameterize(mu, logvar)
#             x_recon    = decoder(z_enc)

#             loss_vae = vae_loss(x_b, x_recon, mu, logvar, beta=args.beta)
#             d_fool   = discriminator(x_b, z_enc)
#             loss_gen = bce_loss(d_fool, ones)

#             # KEY — classification loss shapes z to separate classes
#             logits   = classifier(mu)
#             loss_cls = ce_loss(logits, y_b)

#             loss_total = (loss_vae
#                           + args.lambda_gan * loss_gen
#                           + args.lambda_cls * loss_cls)
#             loss_total.backward()
#             opt_gen.step()

#             ep_vae  += loss_vae.item()
#             ep_disc += loss_d.item()
#             ep_cls  += loss_cls.item()

#         sched.step()

#         # Validation
#         encoder.eval(); classifier.eval()
#         preds_v, true_v = [], []
#         with torch.no_grad():
#             for xv, yv in val_dl:
#                 mu_v, _ = encoder(xv.to(device))
#                 p = classifier(mu_v).argmax(1).cpu().numpy()
#                 preds_v.extend(p)
#                 true_v.extend(yv.numpy())

#         val_acc = accuracy_score(true_v, preds_v) * 100
#         n = len(tr_dl)
#         print(f"  {epoch:>4}  {ep_vae/n:>9.4f}  {ep_disc/n:>7.4f}  "
#               f"{ep_cls/n:>9.4f}  {val_acc:>7.2f}%")

#         if val_acc > best_val_acc:
#             best_val_acc    = val_acc
#             best_enc_state  = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
#             best_dec_state  = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
#             best_disc_state = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
#             best_clf_state  = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

#     # Save
#     torch.save({
#         "encoder":       best_enc_state,
#         "decoder":       best_dec_state,
#         "discriminator": best_disc_state,
#         "input_dim":     input_dim,
#         "latent_dim":    latent_dim,
#     }, SAVE_MODEL_PATH)

#     torch.save({
#         "classifier":  best_clf_state,
#         "latent_dim":  latent_dim,
#         "num_classes": num_classes,
#     }, SAVE_CLF_PATH)

#     meta = {
#         "idx_to_name": idx_to_name,
#         "code_to_idx": code_to_idx,
#         "latent_dim":  latent_dim,
#         "num_classes": num_classes,
#     }
#     with open(SAVE_META_PATH, "wb") as f:
#         pickle.dump(meta, f)

#     print(f"\n[INFO] Best val accuracy : {best_val_acc:.2f}%")
#     print(f"[INFO] Saved → {SAVE_MODEL_PATH}")
#     print(f"[INFO] Saved → {SAVE_CLF_PATH}")
#     print(f"[INFO] Saved → {SAVE_META_PATH}")

#     # ── Final test-set evaluation ──────────────────────────────────
#     print("\n" + "=" * 65)
#     print("  FINAL TEST-SET EVALUATION")
#     print("=" * 65)

#     encoder.load_state_dict(best_enc_state)
#     classifier.load_state_dict(best_clf_state)
#     encoder.eval(); classifier.eval()

#     test_dl = DataLoader(
#         TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
#         batch_size=512, shuffle=False,
#     )
#     preds_t, true_t = [], []
#     with torch.no_grad():
#         for xt, yt in test_dl:
#             mask = yt >= 0
#             if mask.sum() == 0:
#                 continue
#             mu_t, _ = encoder(xt[mask].to(device))
#             p = classifier(mu_t).argmax(1).cpu().numpy()
#             preds_t.extend(p)
#             true_t.extend(yt[mask].numpy())

#     y_true = np.array(true_t)
#     y_pred = np.array(preds_t)
#     class_names = [idx_to_name[i] for i in range(num_classes)]

#     overall_acc = accuracy_score(y_true, y_pred) * 100
#     print(f"\n  Overall Accuracy : {overall_acc:.2f}%\n")

#     print("  Per-Class Classification Report")
#     print("  " + "-" * 60)
#     print(classification_report(y_true, y_pred,
#                                 target_names=class_names,
#                                 zero_division=0))

#     cm = confusion_matrix(y_true, y_pred)
#     print("  Confusion Matrix  (rows=true, cols=predicted)")
#     print("  " + "-" * 60)
#     header = "  {:>25s}  ".format("") + "  ".join(f"{n[:9]:>9}" for n in class_names)
#     print(header)
#     for i, row in enumerate(cm):
#         row_str = "  ".join(f"{v:>9d}" for v in row)
#         print(f"  {class_names[i]:>25s}  {row_str}")

#     print("\n  Per-Class Accuracy")
#     print("  " + "-" * 45)
#     for i, name in enumerate(class_names):
#         mask = y_true == i
#         if mask.sum() == 0:
#             print(f"  {name:>25s}  :  N/A")
#             continue
#         acc_i   = (y_pred[mask] == i).mean() * 100
#         support = int(mask.sum())
#         print(f"  {name:>25s}  :  {acc_i:6.2f}%   (n={support})")

#     print("\n" + "=" * 65)
#     print("  Training and evaluation complete.")
#     print("=" * 65 + "\n")


# # ─────────────────────────────────────────────────────────────────
# # 5. Entry point
# # ─────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--epochs",     type=int,   default=60)
#     p.add_argument("--batch_size", type=int,   default=256)
#     p.add_argument("--latent_dim", type=int,   default=16)
#     p.add_argument("--lr",         type=float, default=1e-3)
#     p.add_argument("--beta",       type=float, default=1.0)
#     p.add_argument("--lambda_gan", type=float, default=0.5)
#     p.add_argument("--lambda_cls", type=float, default=1.5,
#                    help="Weight of classification loss — increase if accuracy is low")
#     p.add_argument("--dropout",    type=float, default=0.3)
#     args = p.parse_args()
#     train(args)




















"""
train_vae_bigan_ann.py
======================
Joint training of VAE-BiGAN + ANN classifier for multiclass classification.

The key idea:
  - A fresh VAE-BiGAN encoder is trained from scratch (recon + KL + GAN losses)
  - SIMULTANEOUSLY, an ANN head on top of z = mu is trained with CrossEntropy
  - Classification gradients flow back through the encoder, so z becomes
    both a good generative latent AND class-discriminative

Compatible with YOUR existing project:
  - Reads from  : test_data.npz                      (already in your folder)
  - Reads from  : saved_state/reverse_label_map.pkl  (already in your folder)
  - y_train / y_test are INTEGER codes — handled correctly here
  - Saves to    : saved_state/vae_bigan_ann_model.pth
                  saved_state/ann_classifier.pth
                  saved_state/ann_multiclass_meta.pkl

Run:
    python train_vae_bigan_ann.py
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────
# Paths  (relative to final_capstone/)
# ─────────────────────────────────────────────────────────────────
DATA_PATH         = "test_data.npz"
REVERSE_LABEL_MAP = "saved_state/reverse_label_map.pkl"

SAVE_MODEL_PATH   = "saved_state/vae_bigan_ann_model.pth"
SAVE_CLF_PATH     = "saved_state/ann_classifier.pth"
SAVE_META_PATH    = "saved_state/ann_multiclass_meta.pkl"


# ─────────────────────────────────────────────────────────────────
# 1. Architecture
# ─────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64),        nn.LeakyReLU(0.2), nn.BatchNorm1d(64),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.LeakyReLU(0.2), nn.BatchNorm1d(64),
            nn.Linear(64, 128),         nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, output_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 64), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(64, 32),                      nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(32, 1),                       nn.Sigmoid(),
        )

    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=1))


class ANNClassifier(nn.Module):
    """Multiclass head on top of latent z."""
    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────────────────────────
# 2. Loss helper
# ─────────────────────────────────────────────────────────────────

def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    recon = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


# ─────────────────────────────────────────────────────────────────
# 3. Data loading
# ─────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found.\n"
            "Make sure you are running this script from inside final_capstone/."
        )
    if not os.path.exists(REVERSE_LABEL_MAP):
        raise FileNotFoundError(
            f"'{REVERSE_LABEL_MAP}' not found.\n"
            "Run train_aae.py or train_bigan.py first to generate it."
        )

    data    = np.load(DATA_PATH)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int64)

    with open(REVERSE_LABEL_MAP, "rb") as f:
        name_to_code = pickle.load(f)           # {name: int_code}
    code_to_name = {v: k for k, v in name_to_code.items()}

    # Remap to consecutive 0..N-1 in case codes have gaps
    unique_codes = np.unique(y_train)
    code_to_idx  = {int(c): i for i, c in enumerate(unique_codes)}
    idx_to_name  = {i: code_to_name.get(int(c), str(c))
                    for i, c in enumerate(unique_codes)}

    y_train_idx = np.array([code_to_idx[int(c)] for c in y_train], dtype=np.int64)
    y_test_idx  = np.array([code_to_idx.get(int(c), -1) for c in y_test], dtype=np.int64)

    num_classes = len(unique_codes)

    print(f"  input_dim   : {X_train.shape[1]}")
    print(f"  num_classes : {num_classes}")
    print(f"  Classes:")
    for idx, name in idx_to_name.items():
        n_tr = int((y_train_idx == idx).sum())
        n_te = int((y_test_idx  == idx).sum())
        print(f"    [{idx}] {name:<30}  train={n_tr:>6}  test={n_te:>5}")
    print(f"  train total : {len(X_train)}   test total : {len(X_test)}\n")

    return (X_train, y_train_idx,
            X_test,  y_test_idx,
            num_classes, idx_to_name, code_to_idx)


# ─────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs("saved_state", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device     : {device}")
    print(f"[INFO] Data file  : {DATA_PATH}")
    print("[INFO] Loading data ...\n")

    (X_train, y_train,
     X_test,  y_test,
     num_classes, idx_to_name, code_to_idx) = load_data()

    input_dim  = X_train.shape[1]
    latent_dim = args.latent_dim

    # Val split from training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    tr_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=512, shuffle=False,
    )

    # Models
    encoder       = Encoder(input_dim, latent_dim).to(device)
    decoder       = Decoder(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim, latent_dim).to(device)
    classifier    = ANNClassifier(latent_dim, num_classes, dropout=args.dropout).to(device)

    opt_gen  = optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(classifier.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    opt_disc = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * 0.5, betas=(0.5, 0.999),
    )

    # ── Class weights: inverse-frequency so minority classes matter ──
    # y_tr contains consecutive 0..N-1 indices
    class_counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1, class_counts)   # avoid /0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # normalise
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"  Class weights (inverse-freq):")
    for i, (name, w) in enumerate(zip([idx_to_name[i] for i in range(num_classes)],
                                       class_weights)):
        print(f"    [{i}] {name:<30} weight={w:.4f}")
    print()

    ce_loss  = nn.CrossEntropyLoss(weight=class_weights_tensor)
    bce_loss = nn.BCELoss()
    sched    = optim.lr_scheduler.StepLR(opt_gen, step_size=20, gamma=0.5)

    best_val_acc = 0.0
    best_enc_state = best_dec_state = best_disc_state = best_clf_state = None

    print(f"[INFO] Training  epochs={args.epochs}  lr={args.lr}  "
          f"batch={args.batch_size}  latent_dim={latent_dim}  "
          f"lambda_cls={args.lambda_cls}\n")
    print(f"  {'Ep':>4}  {'VAE-Loss':>9}  {'D-Loss':>7}  {'Cls-Loss':>9}  {'Val-Acc':>8}")
    print("  " + "-" * 50)

    for epoch in range(1, args.epochs + 1):
        encoder.train(); decoder.train()
        discriminator.train(); classifier.train()

        ep_vae = ep_disc = ep_cls = 0.0

        for x_b, y_b in tr_dl:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            B   = x_b.size(0)
            ones  = torch.ones(B,  1, device=device)
            zeros = torch.zeros(B, 1, device=device)

            # Discriminator update (encoder frozen via no_grad)
            with torch.no_grad():
                mu_d, lv_d = encoder(x_b)
                z_enc_d    = encoder.reparameterize(mu_d, lv_d)
                z_prior    = torch.randn(B, latent_dim, device=device)
                x_fake     = decoder(z_prior)

            opt_disc.zero_grad()
            d_real = discriminator(x_b,    z_enc_d)
            d_fake = discriminator(x_fake, z_prior)
            loss_d = 0.5 * (bce_loss(d_real, ones) + bce_loss(d_fake, zeros))
            loss_d.backward()
            opt_disc.step()

            # Encoder + Decoder + Classifier update
            opt_gen.zero_grad()

            mu, logvar = encoder(x_b)
            z_enc      = encoder.reparameterize(mu, logvar)
            x_recon    = decoder(z_enc)

            loss_vae = vae_loss(x_b, x_recon, mu, logvar, beta=args.beta)
            d_fool   = discriminator(x_b, z_enc)
            loss_gen = bce_loss(d_fool, ones)

            # KEY — classification loss shapes z to separate classes
            logits   = classifier(mu)
            loss_cls = ce_loss(logits, y_b)

            loss_total = (loss_vae
                          + args.lambda_gan * loss_gen
                          + args.lambda_cls * loss_cls)
            loss_total.backward()
            opt_gen.step()

            ep_vae  += loss_vae.item()
            ep_disc += loss_d.item()
            ep_cls  += loss_cls.item()

        sched.step()

        # Validation
        encoder.eval(); classifier.eval()
        preds_v, true_v = [], []
        with torch.no_grad():
            for xv, yv in val_dl:
                mu_v, _ = encoder(xv.to(device))
                p = classifier(mu_v).argmax(1).cpu().numpy()
                preds_v.extend(p)
                true_v.extend(yv.numpy())

        val_acc = accuracy_score(true_v, preds_v) * 100
        n = len(tr_dl)
        print(f"  {epoch:>4}  {ep_vae/n:>9.4f}  {ep_disc/n:>7.4f}  "
              f"{ep_cls/n:>9.4f}  {val_acc:>7.2f}%")

        if val_acc > best_val_acc:
            best_val_acc    = val_acc
            best_enc_state  = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_dec_state  = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            best_disc_state = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
            best_clf_state  = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    # Save
    torch.save({
        "encoder":       best_enc_state,
        "decoder":       best_dec_state,
        "discriminator": best_disc_state,
        "input_dim":     input_dim,
        "latent_dim":    latent_dim,
    }, SAVE_MODEL_PATH)

    torch.save({
        "classifier":  best_clf_state,
        "latent_dim":  latent_dim,
        "num_classes": num_classes,
    }, SAVE_CLF_PATH)

    meta = {
        "idx_to_name": idx_to_name,
        "code_to_idx": code_to_idx,
        "latent_dim":  latent_dim,
        "num_classes": num_classes,
    }
    with open(SAVE_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n[INFO] Best val accuracy : {best_val_acc:.2f}%")
    print(f"[INFO] Saved → {SAVE_MODEL_PATH}")
    print(f"[INFO] Saved → {SAVE_CLF_PATH}")
    print(f"[INFO] Saved → {SAVE_META_PATH}")

    # ── Final test-set evaluation ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL TEST-SET EVALUATION")
    print("=" * 65)

    encoder.load_state_dict(best_enc_state)
    classifier.load_state_dict(best_clf_state)
    encoder.eval(); classifier.eval()

    test_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=512, shuffle=False,
    )
    preds_t, true_t = [], []
    with torch.no_grad():
        for xt, yt in test_dl:
            mask = yt >= 0
            if mask.sum() == 0:
                continue
            mu_t, _ = encoder(xt[mask].to(device))
            p = classifier(mu_t).argmax(1).cpu().numpy()
            preds_t.extend(p)
            true_t.extend(yt[mask].numpy())

    y_true = np.array(true_t)
    y_pred = np.array(preds_t)
    class_names = [idx_to_name[i] for i in range(num_classes)]

    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  Overall Accuracy : {overall_acc:.2f}%\n")

    print("  Per-Class Classification Report")
    print("  " + "-" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix  (rows=true, cols=predicted)")
    print("  " + "-" * 60)
    header = "  {:>25s}  ".format("") + "  ".join(f"{n[:9]:>9}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>9d}" for v in row)
        print(f"  {class_names[i]:>25s}  {row_str}")

    print("\n  Per-Class Accuracy")
    print("  " + "-" * 45)
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            print(f"  {name:>25s}  :  N/A")
            continue
        acc_i   = (y_pred[mask] == i).mean() * 100
        support = int(mask.sum())
        print(f"  {name:>25s}  :  {acc_i:6.2f}%   (n={support})")

    print("\n" + "=" * 65)
    print("  Training and evaluation complete.")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--latent_dim", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--beta",       type=float, default=1.0)
    p.add_argument("--lambda_gan", type=float, default=0.5)
    p.add_argument("--lambda_cls", type=float, default=5.0,
                   help="Weight of classification loss — higher forces encoder to separate classes")
    p.add_argument("--dropout",    type=float, default=0.3)
    args = p.parse_args()
    train(args)
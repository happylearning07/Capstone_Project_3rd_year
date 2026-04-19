"""Train a conditional TVAE generator."""

import os
import pickle
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from gateway.tvae_model import ConditionalTVAE

logger = logging.getLogger("TVAETrainer")
logging.basicConfig(level=logging.INFO,
                    format="%(name)s | %(levelname)s | %(message)s")

WEIGHTS_DIR            = "gateway/weights"
TVAE_WEIGHTS_PATH      = os.path.join(WEIGHTS_DIR, "tvae_weights.pth")
TVAE_LABEL_ENC_PATH    = os.path.join(WEIGHTS_DIR, "tvae_label_encoder.pkl")
TVAE_ID_TO_CODE_PATH   = os.path.join(WEIGHTS_DIR, "tvae_id_to_pipeline_code.pkl")
TVAE_FEATURE_COLS_PATH = os.path.join(WEIGHTS_DIR, "tvae_feature_cols.pkl")
TVAE_SCALER_PATH       = os.path.join(WEIGHTS_DIR, "tvae_scaler.pkl")

PIPELINE_SCALER_PATH       = "saved_state/scaler.pkl"
PIPELINE_FEATURE_COLS_PATH = "saved_state/feature_cols.pkl"
PIPELINE_REVERSE_MAP_PATH  = "saved_state/reverse_label_map.pkl"


def _load_training_data(csv_path: str, n_rows: int):
    from gateway.trainer import load_training_data
    return load_training_data(csv_path, n_rows=n_rows)


def _balance_classes(X: np.ndarray, y: np.ndarray) -> tuple:
    counts    = Counter(y)
    max_count = max(counts.values())
    X_parts, y_parts = [X], [y]
    for cls, cnt in counts.items():
        if cnt < max_count:
            idx      = np.where(y == cls)[0]
            needed   = max_count - cnt
            rep_idx  = np.random.choice(idx, size=needed, replace=True)
            X_parts.append(X[rep_idx])
            y_parts.append(np.full(needed, cls))
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    perm  = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def _per_class_stats(model: ConditionalTVAE, le, n_samples: int = 200) -> dict:
    model.eval()
    stds  = {}
    means = {}
    with torch.no_grad():
        for cid, cname in enumerate(le.classes_):
            lbl = torch.full((n_samples,), cid, dtype=torch.long)
            z   = torch.randn(n_samples, model.latent_dim)
            out = model.decode(z, lbl)
            stds[cname]  = round(float(out.std().item()), 4)
            means[cname] = round(float(out.mean().item()), 4)
    model.train()

    inter_spread  = round(max(means.values()) - min(means.values()), 4)
    min_class_std = round(min(stds.values()), 4)
    return {
        "stds": stds, "means": means,
        "inter_spread": inter_spread,
        "min_class_std": min_class_std,
    }


def _per_class_mae_vs_real(model: ConditionalTVAE,
                            X: np.ndarray, y: np.ndarray,
                            le, device: str) -> dict:
    """Per-class MAE of generated vs real feature means."""
    model.eval()
    maes = {}
    with torch.no_grad():
        for cid, cname in enumerate(le.classes_):
            mask  = y == cid
            n_cls = mask.sum()
            if n_cls < 5:
                maes[cname] = float("nan")
                continue

            # Real mean for this class
            real_mean = X[mask].mean(axis=0)  # (feature_dim,)

            # Generated mean for this class
            lbl   = torch.full((500,), cid, dtype=torch.long, device=device)
            z     = torch.randn(500, model.latent_dim, device=device)
            gen   = model.decode(z, lbl).cpu().numpy()
            gen_mean = gen.mean(axis=0)  # (feature_dim,)

            mae = float(np.abs(gen_mean - real_mean).mean())
            maes[cname] = round(mae, 5)

    model.train()
    return maes


def _pearson_r_vs_real(model: ConditionalTVAE,
                        X: np.ndarray, y: np.ndarray,
                        le, device: str) -> dict:
    """Per-class Pearson correlation between generated and real feature means."""
    model.eval()
    rs = {}
    with torch.no_grad():
        for cid, cname in enumerate(le.classes_):
            mask = y == cid
            if mask.sum() < 5:
                rs[cname] = float("nan")
                continue
            real_mean = X[mask].mean(axis=0)
            lbl  = torch.full((500,), cid, dtype=torch.long, device=device)
            z    = torch.randn(500, model.latent_dim, device=device)
            gen  = model.decode(z, lbl).cpu().numpy()
            gen_mean = gen.mean(axis=0)

            if real_mean.std() < 1e-9 or gen_mean.std() < 1e-9:
                rs[cname] = 0.0
            else:
                r = float(np.corrcoef(real_mean, gen_mean)[0, 1])
                rs[cname] = round(r if not np.isnan(r) else 0.0, 4)
    model.train()
    return rs


def _tvae_loss(x_hat, x, mu, logvar, beta):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl    = -0.5 * torch.sum(
        1 + torch.clamp(logvar, -10, 10) - mu.pow(2) - logvar.exp().clamp(max=1e6)
    ) / x.size(0)
    return recon + beta * kl, recon, kl


def train_tvae(
    csv_path:    str,
    n_rows:      int   = 300_000,
    epochs:      int   = 100,
    batch_size:  int   = 512,
    latent_dim:  int   = 32,
    embed_dim:   int   = 32,
    hidden_dims: tuple = (256, 128),
    lr:          float = 1e-3,
    beta_max:    float = 0.1,
    beta_warmup: float = 0.3,
    save_dir:    str   = WEIGHTS_DIR,
    device:      str   = "cpu",
    log_every:   int   = 5,
) -> ConditionalTVAE:
    """Train Conditional TVAE on IoT-23 data."""
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Loading training data from %s (n_rows=%d)", csv_path, n_rows)
    X, y_int, le, scaler, id_to_code, feature_cols = _load_training_data(
        csv_path, n_rows)

    logger.info("Before balance: %s",
                dict(zip(*np.unique(y_int, return_counts=True))))
    X, y_int = _balance_classes(X, y_int)
    logger.info("After balance:  %s",
                dict(zip(*np.unique(y_int, return_counts=True))))

    num_classes = len(le.classes_)
    feature_dim = X.shape[1]

    logger.info("Real class feature means (reference for MAE metric):")
    for cid, cname in enumerate(le.classes_):
        mask = y_int == cid
        if mask.sum() > 0:
            m = X[mask].mean()
            s = X[mask].std()
            logger.info("  [%d] %-32s global_mean=%.4f  std=%.4f  n=%d",
                        cid, cname, m, s, int(mask.sum()))

    model = ConditionalTVAE(
        feature_dim  = feature_dim,
        num_classes  = num_classes,
        latent_dim   = latent_dim,
        embed_dim    = embed_dim,
        hidden_dims  = hidden_dims,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y_int))
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, drop_last=True)

    warmup_epochs = max(1, int(epochs * beta_warmup))

    logger.info(
        "Training TVAE | epochs=%d | batch=%d | latent=%d | embed=%d | "
        "hidden=%s | lr=%.4f | beta_max=%.3f | "
        "classes=%d | features=%d",
        epochs, batch_size, latent_dim, embed_dim, hidden_dims,
        lr, beta_max, num_classes, feature_dim,
    )

    best_avg_mae = float("inf")

    for epoch in range(epochs):
        model.train()

        beta = beta_max * min(1.0, epoch / max(warmup_epochs, 1))

        total_losses, recon_losses, kl_losses_ep = [], [], []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_hat, mu, logvar = model(x_batch, y_batch)
            loss, recon, kl   = _tvae_loss(x_hat, x_batch, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_losses.append(loss.item())
            recon_losses.append(recon.item())
            kl_losses_ep.append(kl.item())

        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            maes = _per_class_mae_vs_real(model, X, y_int, le, device)
            rs   = _pearson_r_vs_real(model, X, y_int, le, device)
            stats = _per_class_stats(model, le)

            valid_maes = [v for v in maes.values() if not np.isnan(v)]
            avg_mae = np.mean(valid_maes) if valid_maes else float("inf")

            valid_rs = [v for v in rs.values() if not np.isnan(v)]
            avg_r = np.mean(valid_rs) if valid_rs else 0.0

            logger.info(
                "Epoch %3d/%d | Recon=%.5f | KL=%.4f | β=%.4f | "
                "avg_MAE=%.5f | avg_Pearson_r=%.4f",
                epoch + 1, epochs,
                np.mean(recon_losses),
                np.mean(kl_losses_ep),
                beta, avg_mae, avg_r,
            )
            logger.info("  Per-class MAE vs real (lower=better): %s", maes)
            logger.info("  Per-class Pearson r  (higher=better): %s", rs)
            logger.info("  inter_spread=%.4f", stats["inter_spread"])

            if avg_mae < best_avg_mae:
                best_avg_mae = avg_mae
                _save(model, le, scaler, id_to_code, feature_cols,
                      feature_dim, num_classes, latent_dim, embed_dim,
                      hidden_dims, save_dir)
                logger.info("  Best avg MAE=%.5f - saved", best_avg_mae)

    logger.info("\nFinal quality report")
    maes  = _per_class_mae_vs_real(model, X, y_int, le, device)
    rs    = _pearson_r_vs_real(model, X, y_int, le, device)
    stats = _per_class_stats(model, le)

    logger.info("Per-class MAE vs real data:")
    for cls, mae in maes.items():
        bar = "█" * max(0, int((0.05 - mae) / 0.001)) if not np.isnan(mae) else ""
        logger.info("  %-30s MAE=%.5f  %s", cls, mae, bar)

    logger.info("\nPer-class Pearson r (feature emphasis preserved):")
    for cls, r in rs.items():
        bar = "█" * int(max(0, r) * 20) if not np.isnan(r) else ""
        logger.info("  %-30s r=%.4f  [%s]", cls, r, bar)

    logger.info("\ninter_spread=%.4f", stats["inter_spread"])

    _save(model, le, scaler, id_to_code, feature_cols,
          feature_dim, num_classes, latent_dim, embed_dim, hidden_dims, save_dir)
    return model


def _save(model, le, scaler, id_to_code, feature_cols,
          feature_dim, num_classes, latent_dim, embed_dim, hidden_dims, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "state_dict":  model.state_dict(),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "latent_dim":  latent_dim,
        "embed_dim":   embed_dim,
        "hidden_dims": list(hidden_dims),
    }, TVAE_WEIGHTS_PATH)
    for name, obj in [
        (TVAE_LABEL_ENC_PATH,    le),
        (TVAE_ID_TO_CODE_PATH,   id_to_code),
        (TVAE_FEATURE_COLS_PATH, feature_cols),
        (TVAE_SCALER_PATH,       scaler),
    ]:
        with open(name, "wb") as f:
            pickle.dump(obj, f)
    logger.info("TVAE saved to %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional TVAE")
    parser.add_argument("--data",      required=True)
    parser.add_argument("--rows",      type=int,   default=300_000)
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--batch",     type=int,   default=512)
    parser.add_argument("--latent",    type=int,   default=32)
    parser.add_argument("--embed",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--beta-max",  type=float, default=0.1)
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--log-every", type=int,   default=5)
    args = parser.parse_args()

    train_tvae(
        csv_path   = args.data,
        n_rows     = args.rows,
        epochs     = args.epochs,
        batch_size = args.batch,
        latent_dim = args.latent,
        embed_dim  = args.embed,
        lr         = args.lr,
        beta_max   = args.beta_max,
        device     = args.device,
        log_every  = args.log_every,
    )
"""Gateway generator and adversarial sample builder."""

import logging
import numpy as np
import torch
import torch.nn as nn
import os
import pickle

from gateway.gmm_generator import GMMGenerator
from gateway.utils import (
    load_cgan,
    name_to_compact_id,
    inject_feature_noise,
    truncate_sequence,
    partial_latent_noise,
)

logger = logging.getLogger("Gateway")
logging.basicConfig(level=logging.INFO,
                    format="%(name)s | %(levelname)s | %(message)s")

CGAN_WEIGHTS      = "gateway/weights/cgan_weights.pth"
ROBUST_CALIB_PATH = "robust_calibration.pkl"


class Gateway:
    def __init__(self,
                 generator_type: str    = "cgan",
                 use_gan_refiner: bool  = True,
                 gan_blend_alpha: float = 0.1,
                 epsilon: float         = 0.05,
                 benign_label: str      = "Benign",
                 benign_ratio: float    = 0.70):

        self.epsilon         = epsilon
        self.benign_label    = benign_label
        self.benign_ratio    = benign_ratio
        self.gan_blend_alpha = gan_blend_alpha

        # Load robust calibration so FGSM knows the robust model's flip direction
        self._robust_flip = False
        if os.path.exists(ROBUST_CALIB_PATH):
            try:
                with open(ROBUST_CALIB_PATH, 'rb') as f:
                    rob_calib = pickle.load(f)
                self._robust_flip = bool(rob_calib.get('flip', False))
                logger.info("Robust calibration loaded: flip=%s", self._robust_flip)
            except Exception as e:
                logger.warning("Could not load robust_calibration.pkl (%s) - "
                               "defaulting flip=False", e)

        self.gmm            = GMMGenerator.load()
        self.all_classes    = list(self.gmm.gmms.keys())
        self.attack_classes = [c for c in self.all_classes
                               if c != self.benign_label]
        self._pipeline_map  = self.gmm.reverse_map

        gmm_dim = len(self.gmm.feature_cols)

        self.gan_model    = None
        self.label_enc    = None
        self.feature_cols = None

        if use_gan_refiner and generator_type != "none":
            try:
                if generator_type == "cgan":
                    (self.gan_model, self.label_enc, _,
                     self.feature_cols, _) = load_cgan()
                elif generator_type == "tvae":
                    from gateway.load_tvae import load_tvae
                    (self.gan_model, self.label_enc, _,
                     self.feature_cols, _) = load_tvae()
                else:
                    raise ValueError(
                        f"Unknown generator_type '{generator_type}'. "
                        "Use 'cgan', 'tvae', or 'none'."
                    )

                gan_dim = self.gan_model.feature_dim
                if gan_dim != gmm_dim:
                    logger.warning(
                        "Generator refiner DISABLED: dimension mismatch "
                        "(GMM=%d, generator=%d). Falling back to GMM-only.",
                        gmm_dim, gan_dim,
                    )
                    self.gan_model    = None
                    self.label_enc    = None
                    self.feature_cols = None
                else:
                    logger.info(
                        "Generator '%s' loaded - hybrid mode active (dim=%d).",
                        generator_type, gan_dim
                    )
            except Exception as e:
                logger.warning(
                    "Generator '%s' load failed (%s) - GMM-only mode.",
                    generator_type, e
                )
        else:
            logger.info("GMM-only mode.")

        logger.info("Gateway ready | classes=%d | GMM+GAN=%s | blend_α=%.1f | ε=%.3f",
                    len(self.all_classes),
                    self.gan_model is not None,
                    self.gan_blend_alpha,
                    epsilon)

    #  Public API

    def generate(self, mode="stream", n_samples=100,
                 attack_type=None, stage="full",
                 ids_model=None, model_type="bigan"):
        if mode == "stream":
            return self._stream_mode(n_samples)
        elif mode == "attack":
            if attack_type is None:
                attack_type = self.attack_classes[0]
            return self._attack_mode(n_samples, attack_type, stage)
        elif mode == "adversarial":
            if attack_type is None:
                attack_type = self.attack_classes[0]
            return self._adversarial_mode(n_samples, attack_type,
                                          ids_model, model_type)
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

    #  Core generation

    def _generate_class(self, label_name: str, n: int,
                        noise_override=None) -> np.ndarray:
        X_gmm = self.gmm.sample_class(label_name, n)

        if self.gan_model is None or self.gan_blend_alpha == 0.0:
            return X_gmm

        try:
            compact_id = name_to_compact_id(label_name, self.label_enc)
            labels_t   = torch.full((n,), compact_id, dtype=torch.long)

            with torch.no_grad():
                if noise_override is not None:
                    z = noise_override
                else:
                    z_rand = torch.randn(n, self.gan_model.latent_dim)
                    x_hint = torch.FloatTensor(X_gmm)
                    proj   = torch.nn.functional.adaptive_avg_pool1d(
                        x_hint.unsqueeze(0),
                        self.gan_model.latent_dim
                    ).squeeze(0)
                    proj   = (proj - proj.mean()) / (proj.std() + 1e-8)
                    z      = 0.5 * z_rand + 0.5 * proj

                X_gan = self.gan_model.generate(labels_t, z).numpy()

            X_out = (1.0 - self.gan_blend_alpha) * X_gmm + self.gan_blend_alpha * X_gan
            return np.clip(X_out, 0.0, 1.0).astype(np.float32)

        except Exception as e:
            logger.warning("GAN refiner failed for '%s' (%s) - GMM only.", label_name, e)
            return X_gmm

    def _pipeline_code(self, label_name: str) -> int:
        return self._pipeline_map.get(label_name, 0)

    #  Stream mode

    def _stream_mode(self, n_samples: int) -> tuple:
        logger.info("Stream mode | n=%d | benign_ratio=%.0f%%",
                    n_samples, self.benign_ratio * 100)

        n_benign = int(n_samples * self.benign_ratio)
        n_attack = n_samples - n_benign

        X_parts, y_parts, meta = [], [], []

        X_b    = self._generate_class(self.benign_label, n_benign)
        code_b = self._pipeline_code(self.benign_label)
        X_parts.append(X_b)
        y_parts.extend([code_b] * n_benign)
        meta.extend([{'type': self.benign_label, 'label_name': self.benign_label,
                      'mode': 'stream', 'stage': 'stream',
                      'perturbed': False}] * n_benign)

        if self.attack_classes and n_attack > 0:
            per_cls   = n_attack // len(self.attack_classes)
            remainder = n_attack %  len(self.attack_classes)
            for i, atype in enumerate(self.attack_classes):
                n_this = per_cls + (1 if i < remainder else 0)
                if n_this == 0:
                    continue
                X_a  = self._generate_class(atype, n_this)
                code = self._pipeline_code(atype)
                X_parts.append(X_a)
                y_parts.extend([code] * n_this)
                meta.extend([{'type': atype, 'label_name': atype,
                              'mode': 'stream', 'stage': 'stream',
                              'perturbed': False}] * n_this)

        X   = np.vstack(X_parts)
        y   = np.array(y_parts)
        idx = np.random.permutation(len(y))
        return X[idx], y[idx], [meta[i] for i in idx]

    #  Attack mode

    def _attack_mode(self, n_samples, attack_type, stage):
        logger.info("Attack mode | type=%s | stage=%s | n=%d",
                    attack_type, stage, n_samples)

        if stage == "early":
            latent_dim = (self.gan_model.latent_dim
                        if self.gan_model is not None else 64)
            z_partial  = partial_latent_noise(n_samples, latent_dim,
                                              noise_fraction=0.5)
            X = self._generate_class(attack_type, n_samples,
                                     noise_override=z_partial)
            X = inject_feature_noise(X, sigma=0.15)
            X = self._truncate_random_features(X, keep_fraction=0.5)
        else:
            X = self._generate_class(attack_type, n_samples)

        code = self._pipeline_code(attack_type)
        y    = np.full(n_samples, code, dtype=int)
        meta = [{'type': attack_type, 'label_name': attack_type,
                 'mode': 'attack', 'stage': stage,
                 'perturbed': False}] * n_samples
        return X, y, meta

    @staticmethod
    def _truncate_random_features(X: np.ndarray,
                                  keep_fraction: float = 0.5) -> np.ndarray:
        n_samples, n_features = X.shape
        n_keep  = max(1, int(n_features * keep_fraction))
        X_trunc = X.copy()
        for i in range(n_samples):
            zero_idx = np.random.choice(n_features,
                                        size=n_features - n_keep,
                                        replace=False)
            X_trunc[i, zero_idx] = 0.0
        return X_trunc

    #  Adversarial mode

    def _adversarial_mode(self, n_samples, attack_type, ids_model, model_type):
        logger.info("Adversarial mode | type=%s | model=%s | n=%d | ε=%.3f",
                    attack_type, model_type, n_samples, self.epsilon)

        X    = self._generate_class(attack_type, n_samples)
        code = self._pipeline_code(attack_type)

        if ids_model is None:
            logger.warning("ids_model not provided - returning unperturbed samples.")
            y    = np.full(n_samples, code, dtype=int)
            meta = [{'type': attack_type, 'label_name': attack_type,
                     'mode': 'adversarial', 'stage': 'full',
                     'perturbed': False}] * n_samples
            return X, y, meta

        X_adv = self._apply_fgsm(X, ids_model, model_type)
        y     = np.full(n_samples, code, dtype=int)
        meta  = [{'type': attack_type, 'label_name': attack_type,
                  'mode': 'adversarial', 'stage': 'full',
                  'perturbed': True, 'epsilon': self.epsilon}] * n_samples
        return X_adv, y, meta

    #  FGSM - per-model scoring formula aware

    def _apply_fgsm(self, X: np.ndarray, model, model_type: str) -> np.ndarray:
        """
        FGSM evasion derived from each model's actual anomaly scoring formula.

        BiGAN (flip=True, combined score):
            anomaly = 1 - [0.5*(1-d) + 0.5*recon]
            To evade: want anomaly LOW -> want [0.5*(1-d) + 0.5*recon] HIGH
                      -> want d LOW  (so 1-d is high)
                      -> loss = BCE(d, zeros), subtract gradient
            Note: recon also rises when x is perturbed (harder to reconstruct),
            which helps evasion. So attacking d alone is sufficient and consistent.

        Robust VAE-BiGAN (flip=True, d-only score):
            anomaly = 1 - d_score
            To evade: want anomaly LOW -> want d HIGH -> want d -> 1 (benign region)
            -> loss = BCE(d, ones), subtract gradient
            Opposite direction to BiGAN because the scoring formula is different.

        AAE (no flip, RF on latent space):
            Gradient-free RF classifier. Attack the encoder reconstruction loss.
            Lower MSE(decoder(encoder(x)), x) -> sample looks reconstructable
            -> maps to a familiar latent region -> classified as known (benign-like).
            -> loss = MSE(recon, x), subtract gradient
        """
        model.eval()
        bce = nn.BCELoss()
        mse = nn.MSELoss()

        logger.info("FGSM | model_type=%s | ε=%.3f | n=%d",
                    model_type, self.epsilon, len(X))

        X_adv_list = []

        for i in range(len(X)):
            x_t = torch.FloatTensor(X[i:i+1]).requires_grad_(True)

            if model_type == "bigan":
                # BiGAN anomaly = 1 - [0.5*(1-d) + 0.5*recon]
                # Evade: lower anomaly = raise (1-d) term = lower d
                # loss = BCE(d, zeros) -> minimising pushes d -> 0 -> anomaly -> 0
                z_enc = model.encoder(x_t)
                pair  = torch.cat([x_t, z_enc], dim=1)
                d_out = model.discriminator(pair)
                loss  = bce(d_out, torch.zeros(1, 1))

            elif model_type == "aae":
                # No gradient through RF. Attack reconstruction loss instead.
                # Lower MSE -> looks like known traffic -> more likely benign class
                z_enc   = model.encoder(x_t)
                x_recon = model.decoder(z_enc)
                loss    = mse(x_recon, x_t)

            elif model_type == "robust":
                # Robust anomaly = 1 - d_score
                # Evade: lower anomaly = raise d_score -> d -> 1 (benign region)
                # Opposite to BiGAN because scoring uses d directly, not (1-d)
                # loss = BCE(d, ones) -> minimising pushes d -> 1 -> anomaly -> 0
                mu, logvar = model.encode(x_t)
                z_enc      = model.reparameterize(mu, logvar)
                pair       = torch.cat([x_t, z_enc], dim=1)
                d_out      = model.discriminator(pair)
                loss       = bce(d_out, torch.ones(1, 1))

            else:
                raise ValueError(f"Unknown model_type '{model_type}'. "
                                 f"Use 'bigan', 'aae', or 'robust'.")

            loss.backward()

            with torch.no_grad():
                x_adv = x_t - self.epsilon * x_t.grad.sign()
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

            X_adv_list.append(x_adv.detach().numpy())

        return np.vstack(X_adv_list)
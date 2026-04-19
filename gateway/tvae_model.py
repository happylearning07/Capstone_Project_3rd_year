"""Conditional TVAE with class-specific decoder heads."""

import torch
import torch.nn as nn


class ConditionalTVAE(nn.Module):
    """
    Conditional TVAE with shared encoder and per-class decoder heads.

    The shared encoder maps [x | label_embed] -> (mu, logvar).
    The shared decoder trunk maps [z | label_embed] -> hidden.
    Per-class linear heads map hidden -> x_hat with class-specific weights.

    This forces class-specific output WITHOUT requiring different global means,
    which is the correct inductive bias for IoT-23 where classes differ in
    feature covariance structure, not global scale.
    """

    def __init__(self,
                 feature_dim: int,
                 num_classes: int,
                 latent_dim: int = 32,
                 embed_dim: int = 32,
                 hidden_dims: tuple = (256, 128)):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.latent_dim  = latent_dim
        self.embed_dim   = embed_dim

        # Label embedding - shared between encoder and decoder
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        enc_in = feature_dim + embed_dim
        enc_layers = []
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(enc_in, h),
                nn.LayerNorm(h),
                nn.GELU(),
            ]
            enc_in = h
        self.encoder_net = nn.Sequential(*enc_layers)
        self.fc_mu       = nn.Linear(enc_in, latent_dim)
        self.fc_logvar   = nn.Linear(enc_in, latent_dim)

        dec_in = latent_dim + embed_dim
        dec_layers = []
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(dec_in, h),
                nn.LayerNorm(h),
                nn.GELU(),
            ]
            dec_in = h
        self.decoder_trunk = nn.Sequential(*dec_layers)
        self.trunk_out_dim = dec_in

        # Each class gets its own linear layer: trunk_out -> feature_dim
        # This is the key fix: class-specific output projection
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.trunk_out_dim, feature_dim),
                nn.Sigmoid()
            )
            for _ in range(num_classes)
        ])


    def encode(self, x: torch.Tensor,
               labels: torch.LongTensor) -> tuple:
        e   = self.label_embed(labels)
        inp = torch.cat([x, e], dim=1)
        h   = self.encoder_net(inp)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


    def decode(self, z: torch.Tensor,
               labels: torch.LongTensor) -> torch.Tensor:
        e      = self.label_embed(labels)
        inp    = torch.cat([z, e], dim=1)
        hidden = self.decoder_trunk(inp)

        # Route each sample through its class-specific head
        out = torch.zeros(len(z), self.feature_dim,
                          device=z.device, dtype=z.dtype)
        for cid in range(self.num_classes):
            mask = (labels == cid)
            if mask.sum() > 0:
                out[mask] = self.class_heads[cid](hidden[mask])
        return out


    def forward(self, x: torch.Tensor,
                labels: torch.LongTensor) -> tuple:
        mu, logvar = self.encode(x, labels)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decode(z, labels)
        return x_hat, mu, logvar


    def generate(self, labels: torch.LongTensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Generate samples. Matches Gateway's ConditionalGAN.generate() API."""
        self.eval()
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(len(labels), self.latent_dim)
            return self.decode(noise, labels)
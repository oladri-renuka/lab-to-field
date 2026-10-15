"""Conditional VAE: conv encoder/decoder, latent 128, 128x128 RGB output in [0, 1], ~5M params."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))


class ConditionalVAE(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = 128, channels=(32, 64, 128, 256, 256), img_size: int = 128):
        super().__init__()
        self.num_classes, self.latent_dim = num_classes, latent_dim
        self.base = img_size // 2 ** len(channels)
        chans = list(channels)
        enc, prev = [], 3 + num_classes  # one-hot label broadcast as extra input channels
        for c in chans:
            enc.append(_down(prev, c))
            prev = c
        self.encoder = nn.Sequential(*enc)
        flat = chans[-1] * self.base * self.base
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + num_classes, flat)
        rev = chans[::-1]
        self.decoder = nn.Sequential(
            *[_up(a, b) for a, b in zip(rev[:-1], rev[1:])],
            nn.ConvTranspose2d(rev[-1], rev[-1], 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(rev[-1], 3, 3, 1, 1), nn.Sigmoid(),
        )
        self._top = chans[-1]

    def _onehot(self, y):
        return F.one_hot(y, self.num_classes).float()

    def encode(self, x, y):
        h, w = x.shape[-2:]
        cond = self._onehot(y)[:, :, None, None].expand(-1, -1, h, w)
        h = self.encoder(torch.cat([x, cond], 1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, y):
        h = self.fc_dec(torch.cat([z, self._onehot(y)], 1))
        return self.decoder(h.view(-1, self._top, self.base, self.base))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y), mu, logvar

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        return self.decode(torch.randn(len(y), self.latent_dim, device=y.device), y)


def vae_loss(recon, x, mu, logvar, beta: float):
    """Per-image sums averaged over the batch: Gaussian-likelihood recon (MSE) + beta * KL."""
    rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return rec + beta * kl, rec, kl

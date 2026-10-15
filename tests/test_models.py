import torch

from src.models.classifier import build_classifier
from src.models.cvae import ConditionalVAE, vae_loss
from src.models.cyclegan import PatchDiscriminator, ResnetGenerator, count_params


def test_cyclegan_shapes_and_size():
    g, d = ResnetGenerator(), PatchDiscriminator()
    assert 11e6 < count_params(g) < 12e6
    x = torch.randn(1, 3, 256, 256)
    assert g(x).shape == x.shape
    assert d(x).shape == (1, 1, 30, 30)  # 70x70 receptive-field PatchGAN


def test_cvae_shapes_size_and_loss():
    v = ConditionalVAE(num_classes=20)
    assert 4e6 < count_params(v) < 6e6
    x, y = torch.rand(4, 3, 128, 128), torch.tensor([0, 5, 10, 19])
    recon, mu, logvar = v(x, y)
    assert recon.shape == x.shape and mu.shape == (4, 128)
    loss, rec, kl = vae_loss(recon, x, mu, logvar, beta=1.0)
    assert torch.isfinite(loss) and kl >= 0
    assert v.sample(y).shape == x.shape


def test_classifier_head():
    m = build_classifier("efficientnet_b0", 7, pretrained=False)
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 7)

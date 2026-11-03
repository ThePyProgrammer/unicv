"""Tests for the image-conditioned GSDPT Gaussian decoder."""

import torch

from unicv.nn.gsdpt import GSDPT
from unicv.utils.structs import GaussianCloud


def test_gsdpt_predicts_gaussians_for_single_images():
    head = GSDPT(
        embed_dim=32,
        features=32,
        patch_size=16,
        img_size=64,
        sh_degree=0,
    )
    hidden = [torch.randn(2, 17, 32) for _ in range(4)]
    images = torch.randn(2, 3, 64, 64)

    cloud = head(hidden, images)

    assert isinstance(cloud, GaussianCloud)
    assert cloud.xyz.shape == (2, 64 * 64, 3)
    assert cloud.sh_coeffs.shape == (2, 64 * 64, 1, 3)


def test_gsdpt_preserves_batch_and_view_axes():
    head = GSDPT(
        embed_dim=32,
        features=32,
        patch_size=16,
        img_size=64,
        sh_degree=0,
    )
    hidden = [torch.randn(1, 2, 17, 32) for _ in range(4)]
    images = torch.randn(1, 2, 3, 64, 64)

    cloud = head(hidden, images)

    assert cloud.xyz.shape == (1, 2, 64 * 64, 3)

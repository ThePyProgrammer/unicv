"""Tests for GaussianHead (unicv.nn.gaussian)."""

import math

import pytest
import torch

from unicv.nn.gaussian import GaussianHead
from unicv.utils.structs import GaussianCloud


B, C, H, W = 2, 64, 8, 8
N = H * W   # 64 Gaussians per image


def _head(sh_degree: int = 0) -> GaussianHead:
    return GaussianHead(in_channels=C, sh_degree=sh_degree)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_gaussian_head_instantiation():
    head = _head(sh_degree=0)
    assert isinstance(head, GaussianHead)


def test_gaussian_head_invalid_sh_degree():
    with pytest.raises(ValueError, match="sh_degree must be in"):
        GaussianHead(in_channels=C, sh_degree=4)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

def test_gaussian_head_xyz_shape():
    cloud = _head()(torch.zeros(B, C, H, W))
    assert cloud.xyz.shape == (B, N, 3)


def test_gaussian_head_scales_shape():
    cloud = _head()(torch.zeros(B, C, H, W))
    assert cloud.scales.shape == (B, N, 3)


def test_gaussian_head_rotations_shape():
    cloud = _head()(torch.zeros(B, C, H, W))
    assert cloud.rotations.shape == (B, N, 4)


def test_gaussian_head_opacities_shape():
    cloud = _head()(torch.zeros(B, C, H, W))
    assert cloud.opacities.shape == (B, N, 1)


def test_gaussian_head_sh_degree_0_shape():
    cloud = _head(sh_degree=0)(torch.zeros(B, C, H, W))
    assert cloud.sh_coeffs.shape == (B, N, 1, 3)


def test_gaussian_head_sh_degree_3_shape():
    head = GaussianHead(in_channels=C, sh_degree=3)
    cloud = head(torch.zeros(B, C, H, W))
    K = (3 + 1) ** 2   # 16
    assert cloud.sh_coeffs.shape == (B, N, K, 3)


def test_gaussian_head_returns_gaussian_cloud():
    cloud = _head()(torch.zeros(B, C, H, W))
    assert isinstance(cloud, GaussianCloud)


# ---------------------------------------------------------------------------
# Activation constraints
# ---------------------------------------------------------------------------

def test_scales_are_positive():
    cloud = _head()(torch.randn(B, C, H, W))
    assert (cloud.scales > 0).all(), "scales must be strictly positive (softplus)"


def test_opacities_in_unit_interval():
    cloud = _head()(torch.randn(B, C, H, W))
    assert (cloud.opacities >= 0).all()
    assert (cloud.opacities <= 1).all()


def test_rotations_are_unit_quaternions():
    cloud = _head()(torch.randn(B, C, H, W))
    norms = cloud.rotations.norm(dim=-1)   # (B, N)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "rotation quaternions must have unit norm"


# ---------------------------------------------------------------------------
# Derived properties round-trip
# ---------------------------------------------------------------------------

def test_num_gaussians_property():
    cloud = _head()(torch.zeros(B, C, H, W))
    assert cloud.num_gaussians == N


def test_sh_degree_property():
    for deg in (0, 1, 2, 3):
        head = GaussianHead(in_channels=C, sh_degree=deg)
        cloud = head(torch.zeros(B, C, H, W))
        assert cloud.sh_degree == deg

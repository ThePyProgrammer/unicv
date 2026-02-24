"""Tests for SHARP single-image Gaussian splat model (unicv.models.sharp)."""

import torch
import torch.nn as nn
import pytest

from unicv.models.sharp import SHARP, SHARPModel
from unicv.nn.dpt import DPTDecoder
from unicv.nn.gaussian import GaussianHead
from unicv.utils.structs import GaussianCloud
from unicv.utils.types import Modality


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

B, H, W   = 2, 16, 16
N         = H * W      # Gaussians per image
EMBED_DIM = 64
FEATURES  = 32         # decoder output channels == GaussianHead in_channels
PATCH_SIZE = 8
IMG_SIZE   = H
NUM_LEVELS = 4
SH_DEGREE  = 0


class _MockBackbone(nn.Module):
    """Stub ViT backbone: returns NUM_LEVELS fixed hidden states."""

    def __init__(self, embed_dim: int = EMBED_DIM,
                 img_size: int = IMG_SIZE, patch_size: int = PATCH_SIZE) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        n_patches = (img_size // patch_size) ** 2
        self._seq_len = 1 + n_patches

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B_ = x.shape[0]
        return [torch.zeros(B_, self._seq_len, self.embed_dim)] * NUM_LEVELS


def _make_feat_decoder() -> DPTDecoder:
    return DPTDecoder(
        embed_dim=EMBED_DIM,
        features=FEATURES,
        num_layers=NUM_LEVELS,
        patch_size=PATCH_SIZE,
        img_size=IMG_SIZE,
        out_channels=FEATURES,   # feature map, not depth
    )


def _make_sharp(sh_degree: int = SH_DEGREE) -> SHARP:
    return SHARP(
        backbone=_MockBackbone(),
        feature_decoder=_make_feat_decoder(),
        gaussian_head=GaussianHead(in_channels=FEATURES, sh_degree=sh_degree),
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_sharp_instantiation():
    model = _make_sharp()
    assert isinstance(model, SHARP)


# ---------------------------------------------------------------------------
# Output types and shapes
# ---------------------------------------------------------------------------

def test_sharp_returns_gaussian_cloud():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert isinstance(cloud, GaussianCloud)


def test_sharp_xyz_shape():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert cloud.xyz.shape == (B, N, 3)


def test_sharp_scales_shape():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert cloud.scales.shape == (B, N, 3)


def test_sharp_rotations_shape():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert cloud.rotations.shape == (B, N, 4)


def test_sharp_opacities_shape():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert cloud.opacities.shape == (B, N, 1)


def test_sharp_sh_coeffs_shape_degree0():
    model = _make_sharp(sh_degree=0)
    cloud = model(torch.randn(B, 3, H, W))
    assert cloud.sh_coeffs.shape == (B, N, 1, 3)


def test_sharp_sh_coeffs_shape_degree1():
    model = _make_sharp(sh_degree=1)
    cloud = model(torch.randn(B, 3, H, W))
    assert cloud.sh_coeffs.shape == (B, N, 4, 3)


# ---------------------------------------------------------------------------
# Activation constraints (inherited from GaussianHead)
# ---------------------------------------------------------------------------

def test_sharp_scales_positive():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert (cloud.scales > 0).all()


def test_sharp_opacities_in_unit_interval():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert (cloud.opacities >= 0).all() and (cloud.opacities <= 1).all()


def test_sharp_rotations_unit_norm():
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    norms = cloud.rotations.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# Depth-based xyz: z-coordinate should be > 0 (softplus depth)
# ---------------------------------------------------------------------------

def test_sharp_xyz_depth_positive():
    """z-coordinates (camera-space depth) should be > 0 due to softplus."""
    model = _make_sharp()
    cloud = model(torch.randn(B, 3, H, W))
    assert (cloud.xyz[..., 2] > 0).all(), "camera-space depth must be positive"


# ---------------------------------------------------------------------------
# Custom intrinsics
# ---------------------------------------------------------------------------

def test_sharp_accepts_custom_intrinsics():
    model = _make_sharp()
    K = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 50.0
    K[:, 0, 2] = W / 2.0
    K[:, 1, 2] = H / 2.0
    cloud = model(torch.randn(B, 3, H, W), intrinsics=K)
    assert cloud.xyz.shape == (B, N, 3)


# ---------------------------------------------------------------------------
# SHARPModel (VisionModule)
# ---------------------------------------------------------------------------

def test_sharp_model_instantiation():
    model = SHARPModel(_make_sharp())
    assert isinstance(model, SHARPModel)


def test_sharp_model_input_spec():
    model = SHARPModel(_make_sharp())
    assert Modality.RGB in model.input_spec


def test_sharp_model_output_modality():
    model  = SHARPModel(_make_sharp())
    result = model(rgb=torch.randn(B, 3, H, W))
    assert Modality.SPLAT in result


def test_sharp_model_output_is_gaussian_cloud():
    model  = SHARPModel(_make_sharp())
    result = model(rgb=torch.randn(B, 3, H, W))
    assert isinstance(result[Modality.SPLAT], GaussianCloud)


def test_sharp_model_missing_rgb_raises():
    model = SHARPModel(_make_sharp())
    with pytest.raises(KeyError):
        model()

"""Tests for Camera Depth Model (unicv.models.cdm)."""

import torch
import torch.nn as nn
import pytest

from unicv.models.cdm import TokenFusion, CDM, CameraDepthModel
from unicv.nn.dpt import DPTDecoder
from unicv.utils.types import Modality


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

B, C, H, W = 2, 64, 32, 32
EMBED_DIM = 64
NUM_LEVELS = 4
FEATURES = 32
PATCH_SIZE = 8
IMG_SIZE = H   # 32


class _MockBackbone(nn.Module):
    """Returns NUM_LEVELS fixed-shape hidden states (B, 1+N, D)."""

    def __init__(self, embed_dim: int = EMBED_DIM, img_size: int = IMG_SIZE,
                 patch_size: int = PATCH_SIZE) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        n_patches = (img_size // patch_size) ** 2
        self._n = 1 + n_patches   # [CLS] + patches

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.shape[0]
        return [torch.zeros(B, self._n, self.embed_dim)] * NUM_LEVELS


def _make_decoder() -> DPTDecoder:
    return DPTDecoder(
        embed_dim=EMBED_DIM,
        features=FEATURES,
        num_layers=NUM_LEVELS,
        patch_size=PATCH_SIZE,
        img_size=IMG_SIZE,
        out_channels=1,
    )


def _make_cdm() -> CDM:
    return CDM(
        rgb_backbone=_MockBackbone(),
        depth_backbone=_MockBackbone(),
        decoder=_make_decoder(),
        embed_dim=EMBED_DIM,
        num_levels=NUM_LEVELS,
    )


# ---------------------------------------------------------------------------
# TokenFusion tests
# ---------------------------------------------------------------------------

def test_token_fusion_output_shape():
    fusion = TokenFusion(EMBED_DIM)
    rgb   = torch.randn(B, 16, EMBED_DIM)
    depth = torch.randn(B, 16, EMBED_DIM)
    out   = fusion(rgb, depth)
    assert out.shape == (B, 16, EMBED_DIM)


def test_token_fusion_is_differentiable():
    fusion = TokenFusion(EMBED_DIM)
    rgb   = torch.randn(B, 16, EMBED_DIM, requires_grad=True)
    depth = torch.randn(B, 16, EMBED_DIM, requires_grad=True)
    out   = fusion(rgb, depth)
    out.sum().backward()
    assert rgb.grad is not None


def test_token_fusion_identity_at_zero_depth():
    """When depth tokens are all zero, fused ≈ rgb (proj(0)=0)."""
    fusion = TokenFusion(EMBED_DIM)
    nn.init.zeros_(fusion.proj.weight)
    rgb   = torch.randn(B, 16, EMBED_DIM)
    depth = torch.zeros(B, 16, EMBED_DIM)
    out   = fusion(rgb, depth)
    assert torch.allclose(out, rgb)


# ---------------------------------------------------------------------------
# CDM nn.Module
# ---------------------------------------------------------------------------

def test_cdm_instantiation():
    cdm = _make_cdm()
    assert isinstance(cdm, CDM)


def test_cdm_output_shape():
    cdm = _make_cdm()
    rgb   = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 1, H, W)
    out   = cdm(rgb, depth)
    assert out.shape == (B, 1, H, W)


def test_cdm_depth_proj_adapts_single_channel():
    """The internal depth_proj layer should lift (B,1,H,W) → (B,3,H,W)."""
    cdm = _make_cdm()
    assert cdm.depth_proj.in_channels  == 1
    assert cdm.depth_proj.out_channels == 3


def test_cdm_num_fusion_layers():
    cdm = _make_cdm()
    assert len(cdm.fusion_layers) == NUM_LEVELS


def test_cdm_output_dtype_matches_input():
    cdm   = _make_cdm()
    rgb   = torch.randn(B, 3, H, W, dtype=torch.float32)
    depth = torch.randn(B, 1, H, W, dtype=torch.float32)
    out   = cdm(rgb, depth)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# CameraDepthModel (VisionModule)
# ---------------------------------------------------------------------------

def test_camera_depth_model_instantiation():
    model = CameraDepthModel(_make_cdm())
    assert isinstance(model, CameraDepthModel)


def test_camera_depth_model_input_spec():
    model = CameraDepthModel(_make_cdm())
    assert Modality.RGB   in model.input_spec
    assert Modality.DEPTH in model.input_spec


def test_camera_depth_model_output_modality():
    model  = CameraDepthModel(_make_cdm())
    rgb    = torch.randn(B, 3, H, W)
    depth  = torch.randn(B, 1, H, W)
    result = model(rgb=rgb, depth=depth)
    assert Modality.DEPTH in result


def test_camera_depth_model_output_shape():
    model  = CameraDepthModel(_make_cdm())
    rgb    = torch.randn(B, 3, H, W)
    depth  = torch.randn(B, 1, H, W)
    result = model(rgb=rgb, depth=depth)
    assert result[Modality.DEPTH].shape == (B, 1, H, W)


def test_camera_depth_model_missing_depth_raises():
    model = CameraDepthModel(_make_cdm())
    with pytest.raises(KeyError):
        model(rgb=torch.randn(B, 3, H, W))


def test_camera_depth_model_missing_rgb_raises():
    model = CameraDepthModel(_make_cdm())
    with pytest.raises(KeyError):
        model(depth=torch.randn(B, 1, H, W))

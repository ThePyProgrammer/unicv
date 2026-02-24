"""Tests for SimpleRecon multi-view depth estimation (unicv.models.simple_recon)."""

import pytest
import torch

from unicv.models.simple_recon import (
    SimpleEncoder,
    CostVolumeRegularizer,
    SimpleRecon,
    SimpleReconModel,
)
from unicv.nn.cost_volume import PlaneSweepCostVolume
from unicv.utils.types import Modality


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B, C_RGB, H, W = 2, 3, 24, 24
FEAT_C = 16
D      = 8    # number of depth hypotheses
T      = 3    # number of frames (1 ref + 2 src)


def _intrinsics(B: int = B) -> torch.Tensor:
    K = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = float(W)
    K[:, 0, 2] = W / 2.0
    K[:, 1, 2] = H / 2.0
    return K


def _identity_pose(B: int = B) -> torch.Tensor:
    return torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()


def _make_cv() -> PlaneSweepCostVolume:
    return PlaneSweepCostVolume(
        num_depth_hypotheses=D,
        min_depth=0.5,
        max_depth=5.0,
        similarity="ncc",
    )


def _make_model() -> SimpleRecon:
    return SimpleRecon(
        encoder=SimpleEncoder(in_channels=C_RGB, out_channels=FEAT_C),
        cost_volume=_make_cv(),
        regularizer=CostVolumeRegularizer(mid_channels=4),
    )


# ---------------------------------------------------------------------------
# SimpleEncoder
# ---------------------------------------------------------------------------

def test_simple_encoder_output_shape():
    enc = SimpleEncoder(in_channels=3, out_channels=FEAT_C)
    x   = torch.randn(B, 3, H, W)
    out = enc(x)
    assert out.shape == (B, FEAT_C, H // 8, W // 8)


def test_simple_encoder_batch_invariance():
    enc = SimpleEncoder(in_channels=3, out_channels=FEAT_C)
    x1  = torch.randn(1, 3, H, W)
    x2  = torch.randn(3, 3, H, W)
    assert enc(x1).shape[1:] == enc(x2).shape[1:]


def test_simple_encoder_is_differentiable():
    enc = SimpleEncoder(in_channels=3, out_channels=FEAT_C)
    x   = torch.randn(B, 3, H, W, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# CostVolumeRegularizer
# ---------------------------------------------------------------------------

def test_regularizer_output_shape():
    reg    = CostVolumeRegularizer(mid_channels=4)
    volume = torch.randn(B, D, H // 8, W // 8)
    out    = reg(volume)
    assert out.shape == (B, D, H // 8, W // 8)


def test_regularizer_is_differentiable():
    reg    = CostVolumeRegularizer(mid_channels=4)
    volume = torch.randn(B, D, H // 8, W // 8, requires_grad=True)
    reg(volume).sum().backward()
    assert volume.grad is not None


# ---------------------------------------------------------------------------
# SimpleRecon nn.Module
# ---------------------------------------------------------------------------

def test_simple_recon_output_shape():
    model  = _make_model()
    frames = [torch.randn(B, C_RGB, H, W) for _ in range(T)]
    K      = _intrinsics()
    T_id   = _identity_pose()
    out    = model(frames, K, [K] * (T - 1), [T_id] * (T - 1))
    assert out.shape == (B, 1, H, W)


def test_simple_recon_single_source_frame():
    """Pipeline should work with only 1 source frame (T=2)."""
    model  = _make_model()
    frames = [torch.randn(B, C_RGB, H, W), torch.randn(B, C_RGB, H, W)]
    K      = _intrinsics()
    T_id   = _identity_pose()
    out    = model(frames, K, [K], [T_id])
    assert out.shape == (B, 1, H, W)


def test_simple_recon_depth_positive():
    """Softmax + positive depth hypotheses → output should be > 0."""
    model  = _make_model()
    frames = [torch.randn(B, C_RGB, H, W) for _ in range(T)]
    K      = _intrinsics()
    T_id   = _identity_pose()
    out    = model(frames, K, [K] * (T - 1), [T_id] * (T - 1))
    assert (out > 0).all(), "Depth output should be strictly positive"


def test_simple_recon_from_config():
    model = SimpleRecon.from_config(
        feature_channels=FEAT_C,
        num_depth_hypotheses=D,
        min_depth=0.5,
        max_depth=5.0,
    )
    assert isinstance(model, SimpleRecon)


def test_simple_recon_from_config_forward():
    model  = SimpleRecon.from_config(
        feature_channels=FEAT_C,
        num_depth_hypotheses=D,
        min_depth=0.5,
        max_depth=5.0,
    )
    frames = [torch.randn(B, C_RGB, H, W) for _ in range(T)]
    K      = _intrinsics()
    T_id   = _identity_pose()
    out    = model(frames, K, [K] * (T - 1), [T_id] * (T - 1))
    assert out.shape == (B, 1, H, W)


# ---------------------------------------------------------------------------
# SimpleReconModel (VisionModule)
# ---------------------------------------------------------------------------

def test_simple_recon_model_instantiation():
    model = SimpleReconModel(_make_model())
    assert isinstance(model, SimpleReconModel)


def test_simple_recon_model_input_spec():
    model = SimpleReconModel(_make_model())
    assert Modality.RGB in model.input_spec


def test_simple_recon_model_output_modality():
    model  = SimpleReconModel(_make_model())
    frames = [torch.randn(B, C_RGB, H, W) for _ in range(T)]
    result = model(rgb=frames)
    assert Modality.DEPTH in result


def test_simple_recon_model_output_shape():
    model  = SimpleReconModel(_make_model())
    frames = [torch.randn(B, C_RGB, H, W) for _ in range(T)]
    result = model(rgb=frames)
    assert result[Modality.DEPTH].shape == (B, 1, H, W)


def test_simple_recon_model_rejects_scalar_rgb():
    """TEMPORAL input spec should reject a plain tensor (not a list)."""
    model = SimpleReconModel(_make_model())
    with pytest.raises(TypeError):
        model(rgb=torch.randn(B, C_RGB, H, W))

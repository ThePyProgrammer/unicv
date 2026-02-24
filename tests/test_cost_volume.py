"""Tests for PlaneSweepCostVolume (unicv.nn.cost_volume)."""

import pytest
import torch

from unicv.nn.cost_volume import PlaneSweepCostVolume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, C, H, W = 2, 16, 12, 12
D = 8


def _intrinsics(B: int = B) -> torch.Tensor:
    K = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 60.0   # fx, fy
    K[:, 0, 2] = K[:, 1, 2] = 5.5    # cx, cy
    return K


def _identity_pose(B: int = B) -> torch.Tensor:
    return torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()


def _cv(similarity: str = "ncc") -> PlaneSweepCostVolume:
    return PlaneSweepCostVolume(
        num_depth_hypotheses=D,
        min_depth=0.5,
        max_depth=5.0,
        similarity=similarity,
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation_ncc():
    cv = _cv("ncc")
    assert isinstance(cv, PlaneSweepCostVolume)


def test_instantiation_dot():
    cv = _cv("dot")
    assert isinstance(cv, PlaneSweepCostVolume)


def test_invalid_similarity():
    with pytest.raises(ValueError, match="similarity must be"):
        PlaneSweepCostVolume(8, 0.5, 5.0, similarity="l2")


def test_depth_hypotheses_count():
    cv = _cv()
    assert cv.depth_hypotheses.shape == (D,)


def test_log_spacing_monotone():
    cv = PlaneSweepCostVolume(16, 0.1, 10.0, use_log_spacing=True)
    hyps = cv.depth_hypotheses
    assert (hyps[1:] > hyps[:-1]).all(), "log-spaced hypotheses should be monotonically increasing"


def test_linear_spacing_monotone():
    cv = PlaneSweepCostVolume(16, 0.5, 5.0, use_log_spacing=False)
    hyps = cv.depth_hypotheses
    assert (hyps[1:] > hyps[:-1]).all()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape_single_source():
    cv = _cv()
    ref  = torch.randn(B, C, H, W)
    src  = [torch.randn(B, C, H, W)]
    K    = _intrinsics()
    T    = _identity_pose()
    out  = cv(ref, src, K, [K], [T])
    assert out.shape == (B, D, H, W)


def test_output_shape_two_sources():
    cv = _cv()
    ref = torch.randn(B, C, H, W)
    K   = _intrinsics()
    T   = _identity_pose()
    out = cv(ref, [ref, ref], K, [K, K], [T, T])
    assert out.shape == (B, D, H, W)


def test_output_shape_dot_similarity():
    cv = _cv("dot")
    ref = torch.randn(B, C, H, W)
    K   = _intrinsics()
    T   = _identity_pose()
    out = cv(ref, [ref], K, [K], [T])
    assert out.shape == (B, D, H, W)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_identity_pose_produces_high_similarity():
    """Warping the reference onto itself (identity pose) should yield
    maximum NCC ≈ 1.0 at every depth plane for non-constant features."""
    cv = _cv("ncc")
    ref = torch.randn(B, C, H, W)
    K   = _intrinsics()
    T   = _identity_pose()
    out = cv(ref, [ref], K, [K], [T])
    # Interior pixels (away from the zero-padded boundary) should be ≈ 1.
    assert out[:, :, 1:-1, 1:-1].mean().item() > 0.9


def test_zero_source_gives_zero_cost():
    """Zero source features → zero NCC (numerator is zero)."""
    cv = _cv("ncc")
    ref  = torch.randn(B, C, H, W)
    src  = torch.zeros(B, C, H, W)
    K    = _intrinsics()
    T    = _identity_pose()
    out  = cv(ref, [src], K, [K], [T])
    assert (out.abs() < 1e-5).all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_mismatched_src_intrinsics_raises():
    cv = _cv()
    ref = torch.randn(B, C, H, W)
    K   = _intrinsics()
    T   = _identity_pose()
    with pytest.raises(ValueError, match="same length"):
        cv(ref, [ref, ref], K, [K], [T, T])   # 2 src but 1 intrinsics


def test_mismatched_src_poses_raises():
    cv = _cv()
    ref = torch.randn(B, C, H, W)
    K   = _intrinsics()
    T   = _identity_pose()
    with pytest.raises(ValueError, match="same length"):
        cv(ref, [ref], K, [K], [T, T])        # 1 src but 2 poses

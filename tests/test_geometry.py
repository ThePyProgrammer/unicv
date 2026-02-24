"""Tests for backproject_depth and homography_warp (unicv.nn.geometry)."""

import torch

from unicv.nn.geometry import backproject_depth, homography_warp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intrinsics(B: int, fx: float = 100.0, fy: float = 100.0,
                     cx: float = 3.5, cy: float = 3.5) -> torch.Tensor:
    K = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    return K


def _identity_pose(B: int) -> torch.Tensor:
    return torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()


# ---------------------------------------------------------------------------
# backproject_depth
# ---------------------------------------------------------------------------

def test_backproject_output_shape_4d():
    depth = torch.ones(2, 1, 8, 8)
    K = _make_intrinsics(2)
    pts = backproject_depth(depth, K)
    assert pts.shape == (2, 8, 8, 3)


def test_backproject_output_shape_3d_input():
    """Depth given as (B, H, W) should still produce (B, H, W, 3)."""
    depth = torch.ones(2, 8, 8)
    K = _make_intrinsics(2)
    pts = backproject_depth(depth, K)
    assert pts.shape == (2, 8, 8, 3)


def test_backproject_z_equals_depth():
    depth = torch.full((1, 1, 4, 4), 5.0)
    K = _make_intrinsics(1)
    pts = backproject_depth(depth, K)
    assert torch.allclose(pts[..., 2], torch.full((1, 4, 4), 5.0))


def test_backproject_principal_point_maps_to_zero_xy():
    """Pixel at the principal point (cx, cy) should back-project to (0, 0, d)."""
    B, H, W = 1, 8, 8
    cx, cy = 3.0, 3.0
    K = _make_intrinsics(B, fx=50.0, fy=50.0, cx=cx, cy=cy)
    depth = torch.ones(B, 1, H, W)
    pts = backproject_depth(depth, K)

    # Integer pixel (3, 3) → (u=3, v=3) → should give x=0, y=0.
    assert pts[0, 3, 3, 0].abs() < 1e-5   # x
    assert pts[0, 3, 3, 1].abs() < 1e-5   # y
    assert pts[0, 3, 3, 2].item() == pytest.approx(1.0)


def test_backproject_broadcasts_2d_intrinsics():
    """Single (3,3) intrinsics should broadcast over the batch."""
    import pytest
    depth = torch.ones(3, 1, 4, 4)
    K_single = torch.eye(3)
    K_single[0, 0] = K_single[1, 1] = 10.0
    pts = backproject_depth(depth, K_single)
    assert pts.shape == (3, 4, 4, 3)


# ---------------------------------------------------------------------------
# homography_warp
# ---------------------------------------------------------------------------

def test_homography_warp_output_shape():
    B, C, H, W = 2, 16, 8, 8
    src = torch.randn(B, C, H, W)
    K = _make_intrinsics(B)
    T = _identity_pose(B)
    out = homography_warp(src, 1.0, K, K, T)
    assert out.shape == (B, C, H, W)


def test_homography_warp_identity_pose():
    """Identity rotation+translation should reproduce the source features
    (up to bilinear interpolation boundary effects at image edges)."""
    B, C, H, W = 1, 8, 8, 8
    src = torch.randn(B, C, H, W)
    K = _make_intrinsics(B, fx=50.0, fy=50.0, cx=3.5, cy=3.5)
    T = _identity_pose(B)

    out = homography_warp(src, 1.0, K, K, T)

    # Interior pixels should match closely (boundary may be zero-padded).
    assert torch.allclose(out[:, :, 1:-1, 1:-1], src[:, :, 1:-1, 1:-1], atol=1e-4), \
        "Identity warp should reproduce interior pixels"


def test_homography_warp_zero_features():
    """Warping an all-zero source map should always return zeros."""
    B, C, H, W = 2, 4, 8, 8
    src = torch.zeros(B, C, H, W)
    K = _make_intrinsics(B)
    T = _identity_pose(B)
    out = homography_warp(src, 2.0, K, K, T)
    assert (out == 0).all()


def test_homography_warp_scalar_depth():
    B, C, H, W = 1, 4, 6, 6
    src = torch.randn(B, C, H, W)
    K = _make_intrinsics(B)
    T = _identity_pose(B)
    # Both float and tensor depth values should be accepted.
    out_float  = homography_warp(src, 1.0, K, K, T)
    out_tensor = homography_warp(src, torch.tensor(1.0), K, K, T)
    assert torch.allclose(out_float, out_tensor, atol=1e-5)


# Needed for the approx assertion in test_backproject_principal_point
import pytest

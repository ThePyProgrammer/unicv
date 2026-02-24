"""Tests for SparseVoxelTensor, voxelize, devoxelize, and SparseConv3d
(unicv.nn.sparse3d).

SparseConv3d tests are skipped when no backend (spconv / MinkowskiEngine)
is installed.  voxelize / devoxelize are pure-PyTorch and always run.
"""

import pytest
import torch

from unicv.nn.sparse3d import (
    SPARSE3D_BACKEND_AVAILABLE,
    SparseConv3d,
    SparseVoxelTensor,
    devoxelize,
    voxelize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, C, D, H, W = 2, 8, 4, 6, 6


def _dense() -> torch.Tensor:
    return torch.randn(B, C, D, H, W)


def _full_mask() -> torch.Tensor:
    """Mask where every voxel is active."""
    return torch.ones(B, D, H, W, dtype=torch.bool)


def _partial_mask(frac: float = 0.5) -> torch.Tensor:
    """Randomly activate ~frac of voxels (deterministic via manual seed)."""
    gen = torch.Generator()
    gen.manual_seed(0)
    return torch.rand(B, D, H, W, generator=gen) < frac


# ---------------------------------------------------------------------------
# SparseVoxelTensor
# ---------------------------------------------------------------------------

def test_sparse_voxel_tensor_properties():
    N = 10
    sv = SparseVoxelTensor(
        features=torch.zeros(N, C),
        coords=torch.zeros(N, 4, dtype=torch.int),
        batch_size=B,
        spatial_shape=(D, H, W),
    )
    assert sv.num_active == N
    assert sv.num_channels == C


def test_sparse_voxel_tensor_to_device():
    sv = SparseVoxelTensor(
        features=torch.zeros(5, C),
        coords=torch.zeros(5, 4, dtype=torch.int),
        batch_size=1,
        spatial_shape=(D, H, W),
    )
    moved = sv.to("cpu")
    assert moved.features.device.type == "cpu"
    assert moved.coords.device.type == "cpu"


# ---------------------------------------------------------------------------
# voxelize
# ---------------------------------------------------------------------------

def test_voxelize_full_mask_active_count():
    dense = _dense()
    mask  = _full_mask()
    sv = voxelize(dense, mask)
    assert sv.num_active == B * D * H * W


def test_voxelize_partial_mask_active_count():
    dense = _dense()
    mask  = _partial_mask()
    sv = voxelize(dense, mask)
    assert sv.num_active == int(mask.sum().item())


def test_voxelize_output_shapes():
    dense = _dense()
    mask  = _partial_mask()
    sv = voxelize(dense, mask)
    N = sv.num_active
    assert sv.features.shape == (N, C)
    assert sv.coords.shape   == (N, 4)


def test_voxelize_metadata():
    dense = _dense()
    mask  = _full_mask()
    sv = voxelize(dense, mask)
    assert sv.batch_size    == B
    assert sv.spatial_shape == (D, H, W)


def test_voxelize_coord_order():
    """coords[:, 0] should be batch indices in [0, B)."""
    sv = voxelize(_dense(), _partial_mask())
    assert sv.coords[:, 0].min().item() >= 0
    assert sv.coords[:, 0].max().item() <  B


# ---------------------------------------------------------------------------
# devoxelize
# ---------------------------------------------------------------------------

def test_devoxelize_shape():
    sv = voxelize(_dense(), _full_mask())
    dense_out = devoxelize(sv)
    assert dense_out.shape == (B, C, D, H, W)


def test_devoxelize_roundtrip_full_mask():
    """voxelize then devoxelize with a full mask should reproduce dense."""
    dense = _dense()
    sv = voxelize(dense, _full_mask())
    recovered = devoxelize(sv)
    assert torch.allclose(dense, recovered, atol=1e-6)


def test_devoxelize_empty_voxels_use_fill():
    """Inactive voxels should receive the fill_value."""
    dense = _dense()
    mask  = torch.zeros(B, D, H, W, dtype=torch.bool)
    mask[0, 0, 0, 0] = True   # only one active voxel

    sv = voxelize(dense, mask)
    recovered = devoxelize(sv, fill_value=-99.0)

    # All positions except [0,:,0,0,0] must be -99.
    inactive = recovered.clone()
    inactive[0, :, 0, 0, 0] = -99.0   # zero out the one active slot
    assert (inactive == -99.0).all()


def test_voxelize_devoxelize_partial_mask():
    """Active voxel values must survive a partial round-trip."""
    dense = _dense()
    mask  = _partial_mask()
    sv = voxelize(dense, mask)
    recovered = devoxelize(sv, fill_value=0.0)

    # Check only the active positions.
    b, z, y, x = mask.nonzero(as_tuple=True)
    assert torch.allclose(recovered[b, :, z, y, x], dense[b, :, z, y, x], atol=1e-6)


# ---------------------------------------------------------------------------
# SparseConv3d — skipped when no backend is available
# ---------------------------------------------------------------------------

_skip_no_backend = pytest.mark.skipif(
    not SPARSE3D_BACKEND_AVAILABLE,
    reason="No sparse 3-D convolution backend installed (spconv / MinkowskiEngine)",
)


@_skip_no_backend
def test_sparse_conv3d_output_shape():
    conv = SparseConv3d(C, 16, kernel_size=3, padding=1)
    sv = voxelize(_dense(), _partial_mask())
    out = conv(sv)
    assert isinstance(out, SparseVoxelTensor)
    assert out.num_channels == 16


@_skip_no_backend
def test_sparse_conv3d_preserves_batch_size():
    conv = SparseConv3d(C, C, kernel_size=1)
    sv = voxelize(_dense(), _partial_mask())
    out = conv(sv)
    assert out.batch_size == B


# ---------------------------------------------------------------------------
# SparseConv3d — ImportError when no backend
# ---------------------------------------------------------------------------

def test_sparse_conv3d_raises_import_error_without_backend(monkeypatch):
    """When _SPARSE3D_BACKEND is None, constructing SparseConv3d raises
    ImportError with a helpful install message."""
    import unicv.nn.sparse3d as _mod
    original = _mod._SPARSE3D_BACKEND
    monkeypatch.setattr(_mod, "_SPARSE3D_BACKEND", None)
    try:
        with pytest.raises(ImportError, match="spconv"):
            SparseConv3d.__new__(SparseConv3d)   # bypass super().__init__
            # Directly test the constructor with patched state:
            obj = object.__new__(SparseConv3d)
            obj._backend = None
            SparseConv3d.__init__(obj, 8, 8)
    finally:
        monkeypatch.setattr(_mod, "_SPARSE3D_BACKEND", original)

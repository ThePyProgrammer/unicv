"""Sparse 3-D convolution wrapper for voxel-grid models.

This module provides a backend-agnostic interface for sparse 3-D convolutions,
which are required by voxel-based models such as TRELLIS.2.

Two backends are supported, tried in order:

1. **spconv** (``pip install spconv-cuXXX``) — fast CUDA sparse convolutions
   used by many PointPillars / CenterPoint-style detectors.
2. **MinkowskiEngine** (``pip install MinkowskiEngine``) — a general sparse
   tensor library well-suited for 3-D reconstruction tasks.

If neither backend is installed, :class:`SparseConv3d` raises
:exc:`ImportError` at construction time with a helpful installation message.
:func:`voxelize` and :func:`devoxelize` are pure-PyTorch and work regardless
of backend availability.

Coordinate convention (follows spconv):
    ``coords`` is an integer tensor of shape ``(N, 4)`` where each row is
    ``[batch, z, y, x]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Backend detection (lazy — checked once at import time)
# ---------------------------------------------------------------------------

def _detect_backend() -> str | None:
    try:
        import spconv.pytorch  # noqa: F401
        return "spconv"
    except ImportError:
        pass
    try:
        import MinkowskiEngine  # noqa: F401
        return "minkowski"
    except ImportError:
        pass
    return None


_SPARSE3D_BACKEND: str | None = _detect_backend()

#: ``True`` when at least one sparse-conv backend is importable.
SPARSE3D_BACKEND_AVAILABLE: bool = _SPARSE3D_BACKEND is not None


# ---------------------------------------------------------------------------
# Shared sparse tensor dataclass
# ---------------------------------------------------------------------------

@dataclass
class SparseVoxelTensor:
    """Backend-agnostic container for a sparse 3-D feature volume.

    Attributes:
        features:      Active-voxel features, shape ``(N, C)``.
        coords:        Integer voxel coordinates, shape ``(N, 4)``.
                       Column order: ``[batch, z, y, x]``.
        batch_size:    Number of scenes in the batch.
        spatial_shape: Dense grid dimensions ``(D, H, W)``.
    """

    features: torch.Tensor        # (N, C)
    coords: torch.Tensor          # (N, 4)  [batch, z, y, x]
    batch_size: int
    spatial_shape: tuple[int, int, int]   # (D, H, W)

    @property
    def num_active(self) -> int:
        """Number of active (non-empty) voxels."""
        return int(self.features.shape[0])

    @property
    def num_channels(self) -> int:
        """Feature dimensionality ``C``."""
        return int(self.features.shape[1])

    def to(self, *args, **kwargs) -> "SparseVoxelTensor":
        """Move tensors to another device / dtype."""
        return SparseVoxelTensor(
            features=self.features.to(*args, **kwargs),
            coords=self.coords.to(*args, **kwargs),
            batch_size=self.batch_size,
            spatial_shape=self.spatial_shape,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SparseVoxelTensor(N={self.num_active}, C={self.num_channels}, "
            f"spatial_shape={self.spatial_shape}, batch_size={self.batch_size}, "
            f"backend={_SPARSE3D_BACKEND!r})"
        )


# ---------------------------------------------------------------------------
# Pure-PyTorch helpers (no backend required)
# ---------------------------------------------------------------------------

def voxelize(
    dense: torch.Tensor,
    mask: torch.Tensor,
) -> SparseVoxelTensor:
    """Convert a dense feature volume to a :class:`SparseVoxelTensor`.

    Args:
        dense: Dense feature volume, shape ``(B, C, D, H, W)``.
        mask:  Boolean occupancy mask, shape ``(B, D, H, W)``.
               ``True`` where voxels are active.

    Returns:
        :class:`SparseVoxelTensor` with ``N`` active voxels (``N = mask.sum()``).
    """
    B, C, D, H, W = dense.shape

    batch_idx, z_idx, y_idx, x_idx = mask.nonzero(as_tuple=True)   # each (N,)

    features = dense[batch_idx, :, z_idx, y_idx, x_idx]            # (N, C)
    coords   = torch.stack(
        [batch_idx, z_idx, y_idx, x_idx], dim=1
    ).int()                                                          # (N, 4)

    return SparseVoxelTensor(
        features=features,
        coords=coords,
        batch_size=B,
        spatial_shape=(D, H, W),
    )


def devoxelize(
    sparse: SparseVoxelTensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Convert a :class:`SparseVoxelTensor` back to a dense feature volume.

    Args:
        sparse:     Sparse voxel tensor to expand.
        fill_value: Value used to fill empty (inactive) voxels.

    Returns:
        Dense feature volume, shape ``(B, C, D, H, W)``.
    """
    B  = sparse.batch_size
    D, H, W = sparse.spatial_shape
    C  = sparse.num_channels

    dense = torch.full(
        (B, C, D, H, W),
        fill_value,
        dtype=sparse.features.dtype,
        device=sparse.features.device,
    )

    batch_idx = sparse.coords[:, 0].long()
    z_idx     = sparse.coords[:, 1].long()
    y_idx     = sparse.coords[:, 2].long()
    x_idx     = sparse.coords[:, 3].long()

    dense[batch_idx, :, z_idx, y_idx, x_idx] = sparse.features

    return dense


# ---------------------------------------------------------------------------
# SparseConv3d — backend-dispatching sparse convolution
# ---------------------------------------------------------------------------

class SparseConv3d(nn.Module):
    """Sparse 3-D convolution that dispatches to the available backend.

    Accepts and returns :class:`SparseVoxelTensor` objects, handling all
    backend-specific tensor conversions internally.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        kernel_size:  Convolution kernel size (int or 3-tuple).
        stride:       Convolution stride (int or 3-tuple).
        padding:      Zero-padding (int or 3-tuple).
        bias:         Whether to use a learnable bias.

    Raises:
        ImportError: If neither ``spconv`` nor ``MinkowskiEngine`` is installed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self._backend = _SPARSE3D_BACKEND

        if self._backend is None:
            raise ImportError(
                "No sparse 3-D convolution backend found.\n"
                "Install one of:\n"
                "  spconv:          pip install spconv-cu118  "
                "(replace cu118 with your CUDA version)\n"
                "  MinkowskiEngine: pip install MinkowskiEngine"
            )

        if self._backend == "spconv":
            import spconv.pytorch as spconv
            self.conv = spconv.SparseConv3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
        else:
            import MinkowskiEngine as ME
            self.conv = ME.MinkowskiConvolution(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=3,
                bias=bias,
            )

    def forward(self, x: SparseVoxelTensor) -> SparseVoxelTensor:
        """Apply the sparse convolution.

        Args:
            x: Input sparse feature volume.

        Returns:
            Output :class:`SparseVoxelTensor` (spatial shape may change with
            stride > 1).
        """
        if self._backend == "spconv":
            import spconv.pytorch as spconv
            sp = spconv.SparseConvTensor(
                x.features, x.coords, x.spatial_shape, x.batch_size
            )
            out = self.conv(sp)
            return SparseVoxelTensor(
                features=out.features,
                coords=out.indices,
                batch_size=x.batch_size,
                spatial_shape=tuple(out.spatial_shape),
            )
        else:
            import MinkowskiEngine as ME
            sp = ME.SparseTensor(features=x.features, coordinates=x.coords)
            out = self.conv(sp)
            return SparseVoxelTensor(
                features=out.F,
                coords=out.C,
                batch_size=x.batch_size,
                spatial_shape=x.spatial_shape,
            )


__all__ = [
    "SPARSE3D_BACKEND_AVAILABLE",
    "SparseVoxelTensor",
    "SparseConv3d",
    "voxelize",
    "devoxelize",
]

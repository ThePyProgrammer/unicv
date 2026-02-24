"""Gaussian parameter regression head.

Maps a dense feature map ``(B, C, H, W)`` to a :class:`GaussianCloud` holding
per-pixel Gaussian splat parameters with all necessary activation constraints
already applied:

- **xyz**: raw 3-D position offsets (no activation; caller is responsible for
  interpreting the coordinate frame).
- **scales**: ``softplus``-activated to guarantee strict positivity.
- **rotations**: ``F.normalize``-d to unit quaternions ``(w, x, y, z)``.
- **opacities**: ``sigmoid``-activated to ``[0, 1]``.
- **sh_coeffs**: raw values, reshaped to ``(B, N, K, 3)`` where
  ``K = (sh_degree + 1) ** 2``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.utils.structs import GaussianCloud


class GaussianHead(nn.Module):
    """Per-pixel Gaussian parameter regression head.

    Applies a set of independent 1×1 convolutions to a dense feature map and
    returns a :class:`GaussianCloud` where each spatial position of the input
    corresponds to one Gaussian.

    Args:
        in_channels: Number of input feature channels ``C``.
        sh_degree:   Maximum spherical-harmonic degree for the colour
                     representation.  ``K = (sh_degree + 1) ** 2`` coefficients
                     are predicted per colour channel.  Degree 0 = DC colour
                     only (3 values); degree 3 = full 3-DGS colour (48 values).
    """

    def __init__(self, in_channels: int, sh_degree: int = 0) -> None:
        super().__init__()

        if sh_degree < 0 or sh_degree > 3:
            raise ValueError(f"sh_degree must be in [0, 3], got {sh_degree}")

        self.sh_degree = sh_degree
        self._K = (sh_degree + 1) ** 2  # SH terms per colour channel

        def _head(out: int) -> nn.Conv2d:
            return nn.Conv2d(in_channels, out, kernel_size=1, bias=True)

        self.xyz_head      = _head(3)
        self.scale_head    = _head(3)
        self.rotation_head = _head(4)
        self.opacity_head  = _head(1)
        self.sh_head       = _head(self._K * 3)

        # Initialise rotation head to produce near-identity quaternions.
        nn.init.zeros_(self.rotation_head.weight)
        nn.init.constant_(self.rotation_head.bias,  # type: ignore[arg-type]
                          0.0)
        # Bias w-component (index 0) to 1 before normalisation.
        self.rotation_head.bias.data[0] = 1.0      # type: ignore[union-attr]

    def forward(self, features: torch.Tensor) -> GaussianCloud:
        """Regress Gaussian parameters from a dense feature map.

        Args:
            features: Dense feature map, shape ``(B, C, H, W)``.

        Returns:
            A :class:`GaussianCloud` with ``N = H × W`` Gaussians per image.
            All leading batch dimensions are preserved; tensor shapes are
            ``(B, N, ...)``.
        """
        B, _C, H, W = features.shape
        N = H * W

        def _flat(t: torch.Tensor) -> torch.Tensor:
            """Conv output (B, k, H, W) → (B, N, k)."""
            return t.flatten(start_dim=2).permute(0, 2, 1)

        xyz       = _flat(self.xyz_head(features))       # (B, N, 3)
        scales    = _flat(self.scale_head(features))     # (B, N, 3)
        rotations = _flat(self.rotation_head(features))  # (B, N, 4)
        opacities = _flat(self.opacity_head(features))   # (B, N, 1)
        sh_flat   = _flat(self.sh_head(features))        # (B, N, K*3)

        # Apply constraints.
        scales    = F.softplus(scales)
        rotations = F.normalize(rotations, p=2, dim=-1)
        opacities = torch.sigmoid(opacities)
        sh_coeffs = sh_flat.reshape(B, N, self._K, 3)   # (B, N, K, 3)

        return GaussianCloud(
            xyz=xyz,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
        )


__all__ = ["GaussianHead"]

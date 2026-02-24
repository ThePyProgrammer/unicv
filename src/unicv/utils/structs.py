"""Structured output containers for non-tensor scene representations.

These dataclasses are the canonical return types for ``Modality.SPLAT`` and
``Modality.MESH`` outputs from ``VisionModule`` subclasses.  They carry typed,
device-movable tensors with clearly documented axis semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Gaussian splat container
# ---------------------------------------------------------------------------

@dataclass
class GaussianCloud:
    """A set of 3-D Gaussian primitives representing a scene or object.

    All tensors share a common leading shape ``(..., N)`` where ``N`` is the
    number of Gaussians.  Feed-forward models typically produce a batch
    dimension so tensors have shape ``(B, N, ...)``.  Per-scene optimization
    methods (e.g. 3DGS) usually omit the batch dimension: ``(N, ...)``.

    Attributes:
        xyz:       Centre positions in 3-D space.   Shape ``(..., N, 3)``.
        scales:    Per-axis scale (always positive, stored in activation space
                   after ``softplus``).              Shape ``(..., N, 3)``.
        rotations: Unit quaternions ``(w, x, y, z)`` describing orientation.
                                                    Shape ``(..., N, 4)``.
        opacities: Opacity in ``[0, 1]`` (after ``sigmoid``).
                                                    Shape ``(..., N, 1)``.
        sh_coeffs: Spherical-harmonic colour coefficients.
                   ``K = (sh_degree + 1) ** 2`` terms per colour channel.
                                                    Shape ``(..., N, K, 3)``.
    """

    xyz: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    sh_coeffs: torch.Tensor

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians (the ``N`` dimension)."""
        return int(self.xyz.shape[-2])

    @property
    def sh_degree(self) -> int:
        """Maximum spherical-harmonic degree (0, 1, 2, or 3)."""
        K = self.sh_coeffs.shape[-2]
        return int(math.isqrt(K)) - 1

    # ------------------------------------------------------------------
    # Device / dtype movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> "GaussianCloud":
        """Move all tensors to the given device / dtype (same signature as
        ``torch.Tensor.to``)."""
        return GaussianCloud(
            xyz=self.xyz.to(*args, **kwargs),
            scales=self.scales.to(*args, **kwargs),
            rotations=self.rotations.to(*args, **kwargs),
            opacities=self.opacities.to(*args, **kwargs),
            sh_coeffs=self.sh_coeffs.to(*args, **kwargs),
        )

    def detach(self) -> "GaussianCloud":
        """Return a new :class:`GaussianCloud` with all tensors detached."""
        return GaussianCloud(
            xyz=self.xyz.detach(),
            scales=self.scales.detach(),
            rotations=self.rotations.detach(),
            opacities=self.opacities.detach(),
            sh_coeffs=self.sh_coeffs.detach(),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GaussianCloud(N={self.num_gaussians}, "
            f"sh_degree={self.sh_degree}, "
            f"device={self.xyz.device})"
        )


# ---------------------------------------------------------------------------
# Triangle mesh container
# ---------------------------------------------------------------------------

@dataclass
class TriangleMesh:
    """A triangle mesh with optional per-vertex attributes.

    Attributes:
        vertices:        Vertex positions.          Shape ``(V, 3)`` float.
        faces:           Triangle face indices.     Shape ``(F, 3)`` int64.
        vertex_normals:  Unit normals per vertex.   Shape ``(V, 3)`` or ``None``.
        vertex_colors:   RGB colours in ``[0, 1]``. Shape ``(V, 3)`` or ``None``.
    """

    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_normals: torch.Tensor | None = None
    vertex_colors: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def num_vertices(self) -> int:
        """Number of vertices ``V``."""
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        """Number of triangular faces ``F``."""
        return int(self.faces.shape[0])

    # ------------------------------------------------------------------
    # Device / dtype movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> "TriangleMesh":
        """Move all tensors to the given device / dtype."""
        def _maybe(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(*args, **kwargs) if t is not None else None

        return TriangleMesh(
            vertices=self.vertices.to(*args, **kwargs),
            faces=self.faces.to(*args, **kwargs),
            vertex_normals=_maybe(self.vertex_normals),
            vertex_colors=_maybe(self.vertex_colors),
        )

    def detach(self) -> "TriangleMesh":
        """Return a new :class:`TriangleMesh` with all tensors detached."""
        def _maybe(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.detach() if t is not None else None

        return TriangleMesh(
            vertices=self.vertices.detach(),
            faces=self.faces.detach(),
            vertex_normals=_maybe(self.vertex_normals),
            vertex_colors=_maybe(self.vertex_colors),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TriangleMesh(V={self.num_vertices}, F={self.num_faces}, "
            f"device={self.vertices.device})"
        )


__all__ = ["GaussianCloud", "TriangleMesh"]

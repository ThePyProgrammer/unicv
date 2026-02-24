"""Plane-sweep cost volume for multi-view depth estimation.

The :class:`PlaneSweepCostVolume` module builds a ``(B, D, H, W)`` matching
volume by sweeping a set of fronto-parallel depth hypotheses through the
scene.  For each depth candidate ``d`` and each source view, the source
feature map is warped into the reference view using :func:`homography_warp`,
and the per-pixel similarity between the reference and warped features is
computed.  Similarities are averaged across source views, yielding one cost
slice per depth level.

The default similarity measure is **normalised cross-correlation (NCC)**
computed over the channel dimension, which is robust to illumination changes
between views.  An alternative ``"dot"`` mode computes the dot product of
L2-normalised features (cosine similarity), which is faster and differentiable
everywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from unicv.nn.geometry import homography_warp


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _ncc(feat1: torch.Tensor, feat2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalised cross-correlation over the channel dimension.

    Args:
        feat1: ``(B, C, H, W)``
        feat2: ``(B, C, H, W)``

    Returns:
        ``(B, H, W)`` NCC values in ``[-1, 1]``.
    """
    mean1 = feat1.mean(dim=1, keepdim=True)
    mean2 = feat2.mean(dim=1, keepdim=True)
    f1 = feat1 - mean1
    f2 = feat2 - mean2
    numerator   = (f1 * f2).sum(dim=1)                          # (B, H, W)
    denominator = (
        f1.pow(2).sum(dim=1).sqrt() * f2.pow(2).sum(dim=1).sqrt() + eps
    )
    return numerator / denominator                               # (B, H, W)


def _dot(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity (dot product of L2-normalised features).

    Args:
        feat1: ``(B, C, H, W)``
        feat2: ``(B, C, H, W)``

    Returns:
        ``(B, H, W)`` cosine similarity values in ``[-1, 1]``.
    """
    f1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
    f2 = torch.nn.functional.normalize(feat2, p=2, dim=1)
    return (f1 * f2).sum(dim=1)                                  # (B, H, W)


# ---------------------------------------------------------------------------
# PlaneSweepCostVolume
# ---------------------------------------------------------------------------

class PlaneSweepCostVolume(nn.Module):
    """Builds a plane-sweep cost volume from reference and source features.

    Args:
        num_depth_hypotheses: Number of depth planes ``D``.
        min_depth:            Minimum depth hypothesis (inclusive).
        max_depth:            Maximum depth hypothesis (inclusive).
        use_log_spacing:      If ``True``, depth hypotheses are spaced
                              logarithmically; otherwise linearly.  Log spacing
                              gives more candidates near the camera where
                              depth changes rapidly.
        similarity:           Similarity metric.  One of ``"ncc"``
                              (normalised cross-correlation, default) or
                              ``"dot"`` (cosine similarity).

    Forward inputs:
        ref_feats:            Reference feature map.  ``(B, C, H, W)``.
        src_feats_list:       List of source feature maps.  Each ``(B, C, H, W)``.
        ref_intrinsics:       Reference camera intrinsics.  ``(B, 3, 3)``.
        src_intrinsics_list:  List of source intrinsics.  Each ``(B, 3, 3)``.
        ref_to_src_list:      List of ``T_{src←ref}`` transforms.  Each ``(B, 4, 4)``.

    Forward returns:
        Cost volume ``(B, D, H, W)`` where higher values indicate better
        depth-plane matches.
    """

    def __init__(
        self,
        num_depth_hypotheses: int,
        min_depth: float,
        max_depth: float,
        use_log_spacing: bool = False,
        similarity: str = "ncc",
    ) -> None:
        super().__init__()

        if similarity not in ("ncc", "dot"):
            raise ValueError(f"similarity must be 'ncc' or 'dot', got {similarity!r}")

        self.similarity = similarity

        if use_log_spacing:
            depths = torch.exp(
                torch.linspace(
                    math.log(min_depth), math.log(max_depth), num_depth_hypotheses
                )
            )
        else:
            depths = torch.linspace(min_depth, max_depth, num_depth_hypotheses)

        self.register_buffer("depth_hypotheses", depths)   # (D,)

    def forward(
        self,
        ref_feats: torch.Tensor,
        src_feats_list: list[torch.Tensor],
        ref_intrinsics: torch.Tensor,
        src_intrinsics_list: list[torch.Tensor],
        ref_to_src_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the cost volume.

        Args:
            ref_feats:            ``(B, C, H, W)``
            src_feats_list:       List of ``S`` tensors, each ``(B, C, H, W)``
            ref_intrinsics:       ``(B, 3, 3)`` or ``(3, 3)``
            src_intrinsics_list:  List of ``S`` tensors, each ``(B, 3, 3)`` or ``(3, 3)``
            ref_to_src_list:      List of ``S`` tensors, each ``(B, 4, 4)``

        Returns:
            ``(B, D, H, W)`` cost volume.
        """
        if len(src_feats_list) != len(src_intrinsics_list):
            raise ValueError(
                "src_feats_list and src_intrinsics_list must have the same length"
            )
        if len(src_feats_list) != len(ref_to_src_list):
            raise ValueError(
                "src_feats_list and ref_to_src_list must have the same length"
            )

        sim_fn = _ncc if self.similarity == "ncc" else _dot

        B, _C, H, W = ref_feats.shape
        depth_hyps: torch.Tensor = self.depth_hypotheses  # type: ignore[assignment]
        D = depth_hyps.shape[0]

        cost_volume = torch.zeros(
            B, D, H, W, device=ref_feats.device, dtype=ref_feats.dtype
        )

        for d_idx in range(D):
            d_val = depth_hyps[d_idx].item()
            view_sims: list[torch.Tensor] = []

            for src_feats, src_K, T in zip(
                src_feats_list, src_intrinsics_list, ref_to_src_list
            ):
                warped = homography_warp(src_feats, d_val, ref_intrinsics, src_K, T)
                view_sims.append(sim_fn(ref_feats, warped))  # (B, H, W)

            # Average similarity across source views.
            cost_volume[:, d_idx] = torch.stack(view_sims, dim=0).mean(dim=0)

        return cost_volume


__all__ = ["PlaneSweepCostVolume"]

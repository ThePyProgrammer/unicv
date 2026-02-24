"""SimpleRecon multi-view depth estimation model.

A lightweight plane-sweep stereo network:

1. A small **2-D CNN encoder** extracts per-frame features (1/4 spatial).
2. A **plane-sweep cost volume** (``unicv.nn.PlaneSweepCostVolume``) warps
   source features to the reference camera frustum and measures per-depth
   NCC similarity, producing a ``(B, D, h, w)`` matching volume.
3. A **3-D CNN regularizer** applies spatially-aware smoothing across depth
   and spatial dimensions to reduce photometric noise.
4. **Softmax + weighted sum** over the depth dimension yields an expected
   depth map; it is upsampled to the full input resolution.

Architecture reference
-----------------------
    Sayed et al. – SimpleRecon: 3D Reconstruction Without 3D Convolutions
    https://nianticlabs.github.io/simplerecon/

Note on camera parameters
--------------------------
Camera intrinsics ``K`` and relative poses ``T_{src←ref}`` are required by
the cost volume but are *not* ``Modality`` values in the UniCV type system.
The raw ``SimpleRecon`` module accepts them explicitly.  The
``SimpleReconModel`` VisionModule wrapper defaults to identity camera
parameters (identity intrinsics from image size, identity poses) so that the
interface works without calibration data; for real usage call the nn.Module
directly.

VisionModule spec
-----------------
    Inputs:  Modality.RGB (TEMPORAL): list of ``(B, 3, H, W)`` frames
    Outputs: Modality.DEPTH          : depth map ``(B, 1, H, W)``
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.models.base import VisionModule
from unicv.nn.cost_volume import PlaneSweepCostVolume
from unicv.utils.types import InputForm, Modality


# ---------------------------------------------------------------------------
# Lightweight 2-D feature encoder
# ---------------------------------------------------------------------------

class SimpleEncoder(nn.Module):
    """Lightweight CNN that encodes a single RGB frame into a feature map.

    Three stride-2 convolutions reduce spatial resolution by 8×.  The
    encoder is intentionally shallow so that cost-volume construction stays
    memory-efficient.

    Args:
        in_channels:  Input channels (3 for RGB).
        out_channels: Feature channels in the output map.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        mid = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single frame.

        Args:
            x: ``(B, 3, H, W)``

        Returns:
            Feature map ``(B, out_channels, H//8, W//8)``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# 3-D cost-volume regularizer
# ---------------------------------------------------------------------------

class CostVolumeRegularizer(nn.Module):
    """Small 3-D CNN that regularises a ``(B, D, H, W)`` cost volume.

    The volume is treated as a 5-D tensor ``(B, 1, D, H, W)`` so that
    standard ``Conv3d`` can be applied along all three spatial+depth axes.
    Two residual-style convolutions smooth photometric noise and fill in
    textureless regions.

    Args:
        mid_channels: Internal feature channels for the 3-D convs.
    """

    def __init__(self, mid_channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Regularise the cost volume.

        Args:
            volume: ``(B, D, H, W)`` matching similarity volume.

        Returns:
            Regularised volume, same shape ``(B, D, H, W)``.
        """
        B, D, H, W = volume.shape
        x = volume.unsqueeze(1)                   # (B, 1, D, H, W)
        x = self.net(x)                           # (B, 1, D, H, W)
        return x.squeeze(1)                       # (B, D, H, W)


# ---------------------------------------------------------------------------
# SimpleRecon nn.Module
# ---------------------------------------------------------------------------

class SimpleRecon(nn.Module):
    """Plane-sweep stereo depth estimator.

    Args:
        encoder:      Per-frame 2-D feature extractor.
        cost_volume:  ``PlaneSweepCostVolume`` that defines depth hypotheses
                      and builds the ``(B, D, H, W)`` similarity volume.
        regularizer:  3-D CNN that smooths the volume before soft-argmin.
    """

    def __init__(
        self,
        encoder: SimpleEncoder,
        cost_volume: PlaneSweepCostVolume,
        regularizer: CostVolumeRegularizer,
    ) -> None:
        super().__init__()
        self.encoder     = encoder
        self.cost_volume = cost_volume
        self.regularizer = regularizer

    def forward(
        self,
        frames: List[torch.Tensor],
        intrinsics: torch.Tensor,
        src_intrinsics: List[torch.Tensor],
        src_poses: List[torch.Tensor],
    ) -> torch.Tensor:
        """Estimate a depth map for the reference (first) frame.

        Args:
            frames:          ``T`` frames, each ``(B, 3, H, W)``.  The first
                             frame is the reference; the rest are sources.
            intrinsics:      Reference camera intrinsics ``(B, 3, 3)``.
            src_intrinsics:  Per-source intrinsics, list of ``(B, 3, 3)``.
            src_poses:       Relative poses ``T_{src←ref}``, list of ``(B, 4, 4)``.

        Returns:
            Depth map ``(B, 1, H, W)`` upsampled to the input resolution.
        """
        H_in, W_in = frames[0].shape[-2:]

        ref_feats  = self.encoder(frames[0])
        src_feats  = [self.encoder(f) for f in frames[1:]]

        volume = self.cost_volume(
            ref_feats,
            src_feats,
            intrinsics,
            src_intrinsics,
            src_poses,
        )                                         # (B, D, h, w)

        volume = self.regularizer(volume)         # (B, D, h, w)

        # Softmax + expected depth (soft-argmin over depth hypotheses).
        probs      = volume.softmax(dim=1)        # (B, D, h, w)
        hyps       = self.cost_volume.depth_hypotheses.to(probs)   # (D,)
        depth      = (probs * hyps[None, :, None, None]).sum(dim=1, keepdim=True)  # (B, 1, h, w)

        # Upsample to input resolution.
        if depth.shape[-2:] != (H_in, W_in):
            depth = F.interpolate(depth, size=(H_in, W_in), mode="bilinear", align_corners=True)
        return depth

    @classmethod
    def from_config(
        cls,
        feature_channels: int = 32,
        num_depth_hypotheses: int = 64,
        min_depth: float = 0.25,
        max_depth: float = 10.0,
        use_log_spacing: bool = True,
        regularizer_mid: int = 8,
    ) -> "SimpleRecon":
        """Build a ``SimpleRecon`` from common hyper-parameters.

        Args:
            feature_channels:     Encoder output channels.
            num_depth_hypotheses: Number of depth planes ``D``.
            min_depth:            Minimum depth hypothesis (metres).
            max_depth:            Maximum depth hypothesis (metres).
            use_log_spacing:      Log-space depth hypotheses (recommended).
            regularizer_mid:      Internal channels for the 3-D regularizer.

        Returns:
            An initialised ``SimpleRecon`` model.
        """
        encoder = SimpleEncoder(in_channels=3, out_channels=feature_channels)
        cv      = PlaneSweepCostVolume(
            num_depth_hypotheses=num_depth_hypotheses,
            min_depth=min_depth,
            max_depth=max_depth,
            use_log_spacing=use_log_spacing,
            similarity="ncc",
        )
        reg     = CostVolumeRegularizer(mid_channels=regularizer_mid)
        return cls(encoder=encoder, cost_volume=cv, regularizer=reg)


# ---------------------------------------------------------------------------
# VisionModule wrapper
# ---------------------------------------------------------------------------

def _default_intrinsics(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Build a plausible pinhole intrinsics matrix from image dimensions."""
    K = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    f = float(max(H, W))
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = W / 2.0
    K[:, 1, 2] = H / 2.0
    return K


class SimpleReconModel(VisionModule):
    """UniCV ``VisionModule`` wrapper around ``SimpleRecon``.

    Camera intrinsics and relative poses are estimated / assumed to be
    identity when not supplied.  For real calibrated data, use the
    underlying ``SimpleRecon`` module directly.

    Inputs:
        rgb (Modality.RGB, InputForm.TEMPORAL):
            Temporal sequence of frames, list of ``(B, 3, H, W)`` tensors.
            The first frame is the reference view.

    Outputs:
        Modality.DEPTH: Estimated depth map ``(B, 1, H, W)`` for the
            reference frame.
    """

    input_spec: dict[Modality, InputForm] = {Modality.RGB: InputForm.TEMPORAL}
    output_modalities: list[Modality] = [Modality.DEPTH]

    def __init__(self, net: SimpleRecon) -> None:
        """Initialise SimpleReconModel.

        Args:
            net: Pre-constructed ``SimpleRecon`` instance.
        """
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        """Run depth estimation with default identity camera parameters.

        Args:
            **inputs: Must contain key ``"rgb"`` with a list of frame tensors.

        Returns:
            ``{Modality.DEPTH: depth_tensor}``
        """
        frames: List[torch.Tensor] = inputs[Modality.RGB.value]
        B, _, H, W = frames[0].shape
        device = frames[0].device

        # Build default identity camera parameters.
        K    = _default_intrinsics(B, H, W, device)
        T_id = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        n_src = len(frames) - 1
        src_K = [K] * n_src
        src_T = [T_id] * n_src

        depth = self.net(frames, K, src_K, src_T)
        return {Modality.DEPTH: depth}


__all__ = [
    "SimpleEncoder",
    "CostVolumeRegularizer",
    "SimpleRecon",
    "SimpleReconModel",
]

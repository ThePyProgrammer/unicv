"""SHARP – Single-image Gaussian splat prediction model.

A fully feed-forward model that converts a single RGB image into a dense
3-D Gaussian splat in under one second on a standard GPU.

Architecture
------------
1. **ViT backbone** (any callable returning ``List[Tensor(B, N, D)]``) encodes
   the image into multi-level patch-token sequences.
2. **DPT feature decoder** reassembles the tokens into a dense feature map
   ``(B, F, H, W)`` at full (or half) resolution.
3. **Depth head** – a shallow CNN predicts a per-pixel depth ``(B, 1, H, W)``
   using softplus activation (depth > 0).
4. **GaussianHead** reads the feature map and regresses per-pixel Gaussian
   parameters (scales, rotations, opacities, SH coefficients).
5. **Backprojection** – predicted depth + camera intrinsics → 3-D positions
   ``xyz`` in camera space; replaces the GaussianHead's raw xyz output.

Paper reference
---------------
    SHARP: Single-image 3D Gaussian reconstruction
    https://apple.github.io/ml-sharp/  (arXiv: 2512.10685)

VisionModule spec
-----------------
    Inputs:  Modality.RGB (SINGLE): ``(B, 3, H, W)``
    Outputs: Modality.SPLAT        : ``GaussianCloud`` with ``N = H × W``
                                     Gaussians per image

Camera intrinsics
-----------------
The ``SHARP`` nn.Module accepts an optional ``intrinsics`` argument
``(B, 3, 3)``.  When omitted a default pinhole matrix (focal = max(H,W),
principal point at image centre) is used.  The ``SHARPModel`` VisionModule
always uses this default; callers requiring precise metric scale should
invoke ``SHARP.forward()`` directly with calibrated intrinsics.
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.models.base import VisionModule
from unicv.nn.gaussian import GaussianHead
from unicv.nn.geometry import backproject_depth
from unicv.nn.dpt import DPTDecoder
from unicv.utils.structs import GaussianCloud
from unicv.utils.types import InputForm, Modality


# ---------------------------------------------------------------------------
# SHARP nn.Module
# ---------------------------------------------------------------------------

class SHARP(nn.Module):
    """Single-image Gaussian-splat regression network.

    Args:
        backbone:         ViT backbone returning
                          ``List[Tensor(B, 1+extra+N, embed_dim)]``.
        feature_decoder:  ``DPTDecoder`` with ``out_channels=features``.
        gaussian_head:    ``GaussianHead(in_channels=features, sh_degree=…)``
                          — regresses scales, rotations, opacities, sh_coeffs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_decoder: DPTDecoder,
        gaussian_head: GaussianHead,
    ) -> None:
        super().__init__()
        self.backbone         = backbone
        self.feature_decoder  = feature_decoder
        self.gaussian_head    = gaussian_head

        # Infer the decoder output width from the head's in_channels.
        features = gaussian_head.xyz_head.in_channels   # type: ignore[attr-defined]

        # Shallow depth head: features → (B, 1, H, W) with depth > 0.
        self.depth_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 1, kernel_size=1, bias=True),
            nn.Softplus(),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
    ) -> GaussianCloud:
        """Predict a 3-D Gaussian splat for the input image.

        Args:
            rgb:        ``(B, 3, H, W)`` RGB image.
            intrinsics: Optional camera intrinsics ``(B, 3, 3)``.
                        When ``None``, defaults to a pinhole matrix with
                        focal = max(H, W) and principal point at centre.

        Returns:
            ``GaussianCloud`` with ``N = H × W`` Gaussians per image.
        """
        B, _, H, W = rgb.shape

        # --- Encode + decode to dense feature map ---
        hidden_states: List[torch.Tensor] = self.backbone(rgb)
        feat_map: torch.Tensor            = self.feature_decoder(hidden_states)  # (B, F, H', W')

        if feat_map.shape[-2:] != (H, W):
            feat_map = F.interpolate(feat_map, size=(H, W), mode="bilinear", align_corners=True)

        # --- Gaussian parameters (scales, rotations, opacities, sh_coeffs) ---
        cloud: GaussianCloud = self.gaussian_head(feat_map)

        # --- Depth → 3-D positions via backprojection ---
        depth: torch.Tensor = self.depth_head(feat_map)   # (B, 1, H, W)

        if intrinsics is None:
            f  = float(max(H, W))
            K  = torch.eye(3, device=rgb.device).unsqueeze(0).expand(B, -1, -1).clone()
            K[:, 0, 0] = f
            K[:, 1, 1] = f
            K[:, 0, 2] = W / 2.0
            K[:, 1, 2] = H / 2.0
        else:
            K = intrinsics

        pts = backproject_depth(depth, K)       # (B, H, W, 3)
        xyz = pts.reshape(B, H * W, 3)          # (B, N, 3)

        # Return a new GaussianCloud with 3-D positions from backprojection.
        return GaussianCloud(
            xyz=xyz,
            scales=cloud.scales,
            rotations=cloud.rotations,
            opacities=cloud.opacities,
            sh_coeffs=cloud.sh_coeffs,
        )

    @classmethod
    def from_config(
        cls,
        variant: str = "vit_l",
        img_size: int = 518,
        features: int = 256,
        patch_size: int = 14,
        sh_degree: int = 0,
        pretrained: bool = True,
        num_register_tokens: int = 0,
    ) -> "SHARP":
        """Build SHARP from high-level configuration.

        Args:
            variant:             DINOv2 size (``"vit_s"``/``"vit_b"``/
                                 ``"vit_l"``/``"vit_g"``).
            img_size:            Square input resolution.
            features:            Decoder (and GaussianHead) channel width.
            patch_size:          ViT patch size (14 for all DINOv2 variants).
            sh_degree:           Spherical-harmonic degree in [0, 3].
            pretrained:          Load DINOv2 pretrained weights.
            num_register_tokens: Register tokens (0 for standard DINOv2).

        Returns:
            Initialised ``SHARP`` model.
        """
        from unicv.models.depth_anything_3.model import DINOv2Backbone

        backbone   = DINOv2Backbone(variant, pretrained, num_register_tokens=num_register_tokens)
        embed_dim  = backbone.embed_dim
        num_levels = len(backbone.hook_layer_ids)

        feat_decoder = DPTDecoder(
            embed_dim=embed_dim,
            features=features,
            num_layers=num_levels,
            patch_size=patch_size,
            img_size=img_size,
            num_register_tokens=num_register_tokens,
            out_channels=features,   # feature map, not a single depth channel
        )
        gauss_head = GaussianHead(in_channels=features, sh_degree=sh_degree)

        return cls(
            backbone=backbone,
            feature_decoder=feat_decoder,
            gaussian_head=gauss_head,
        )


# ---------------------------------------------------------------------------
# VisionModule wrapper
# ---------------------------------------------------------------------------

class SHARPModel(VisionModule):
    """UniCV ``VisionModule`` wrapper around ``SHARP``.

    Inputs:
        rgb (Modality.RGB, InputForm.SINGLE): ``(B, 3, H, W)``

    Outputs:
        Modality.SPLAT: ``GaussianCloud`` with ``N = H × W`` Gaussians.
    """

    input_spec: dict[Modality, InputForm] = {Modality.RGB: InputForm.SINGLE}
    output_modalities: list[Modality] = [Modality.SPLAT]

    def __init__(self, net: SHARP) -> None:
        """Initialise SHARPModel.

        Args:
            net: A pre-constructed ``SHARP`` instance.
        """
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        """Run Gaussian-splat prediction with default intrinsics.

        Args:
            **inputs: Must contain key ``"rgb"`` with a tensor value.

        Returns:
            ``{Modality.SPLAT: GaussianCloud}``
        """
        rgb: torch.Tensor = inputs[Modality.RGB.value]
        cloud = self.net(rgb)   # intrinsics=None → default estimate
        return {Modality.SPLAT: cloud}


__all__ = ["SHARP", "SHARPModel"]

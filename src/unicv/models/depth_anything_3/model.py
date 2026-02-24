"""Depth Anything 3 depth estimation model.

Based on:
  ByteDance-Seed - Depth Anything 3 (2025)
  https://github.com/ByteDance-Seed/Depth-Anything-3

Architecture
------------
- **Backbone**: DINOv2 (ViT-L/14) loaded via ``torch.hub``.  The backbone
  exposes 4 intermediate hidden states for use by the decoder.
- **Decoder**: ``DPTDecoder`` from ``unicv.nn.dpt`` – reassembles patch tokens
  into multi-scale spatial maps and progressively fuses them to full resolution.
- **VisionModule wrapper**: ``DepthAnything3Model`` implements the standard
  ``VisionModule`` interface (RGB-in → DEPTH-out).
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.models.base import VisionModule
from unicv.nn.dpt import DPTDecoder
from unicv.utils.types import InputForm, Modality


# ---------------------------------------------------------------------------
# DINOv2 backbone wrapper
# ---------------------------------------------------------------------------

class DINOv2Backbone(nn.Module):
    """Thin wrapper around a ``torch.hub`` DINOv2 model.

    Returns the intermediate hidden states from a configurable set of
    transformer layers so that the DPT decoder can reassemble them.
    """

    # Map config name → hub model name.
    _HUB_NAMES = {
        "vit_s": "dinov2_vits14",
        "vit_b": "dinov2_vitb14",
        "vit_l": "dinov2_vitl14",
        "vit_g": "dinov2_vitg14",
    }
    # embed_dim for each variant.
    _EMBED_DIMS = {
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_g": 1536,
    }

    def __init__(
        self,
        variant: str = "vit_l",
        pretrained: bool = True,
        hook_layer_ids: Optional[List[int]] = None,
        num_register_tokens: int = 0,
    ):
        """Initialise DINOv2Backbone.

        Args:
            variant: Which DINOv2 size to use. One of ``"vit_s"``, ``"vit_b"``,
                ``"vit_l"``, ``"vit_g"``.
            pretrained: Whether to load pretrained weights from ``torch.hub``.
            hook_layer_ids: Indices of transformer blocks whose output should
                be returned.  Defaults to the last four layers.
            num_register_tokens: Number of register tokens used by the model
                (0 for standard DINOv2, 4 for DINOv2-reg).
        """
        super().__init__()

        hub_name = self._HUB_NAMES[variant]
        self.embed_dim: int = self._EMBED_DIMS[variant]
        self.num_register_tokens = num_register_tokens

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=pretrained,
        )

        total_layers = len(self.model.blocks)
        if hook_layer_ids is None:
            # Default: sample 4 layers evenly starting from 1/4 depth.
            step = total_layers // 4
            hook_layer_ids = [
                total_layers // 4 - 1,
                total_layers // 2 - 1,
                3 * total_layers // 4 - 1,
                total_layers - 1,
            ]

        self.hook_layer_ids: List[int] = hook_layer_ids
        self._features: dict[int, torch.Tensor] = {}

        # Register forward hooks on the selected transformer blocks.
        for layer_id in self.hook_layer_ids:
            self.model.blocks[layer_id].register_forward_hook(
                self._make_hook(layer_id)
            )

    def _make_hook(self, layer_id: int):
        def _hook(module, input, output):
            self._features[layer_id] = output
        return _hook

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward through DINOv2 and collect intermediate hidden states.

        Args:
            x: Input images, shape ``(B, 3, H, W)``.

        Returns:
            List of hidden-state tensors in ``hook_layer_ids`` order, each
            shape ``(B, 1 + num_register_tokens + N, D)``.
        """
        self._features.clear()
        _ = self.model(x)
        return [self._features[lid] for lid in self.hook_layer_ids]


# ---------------------------------------------------------------------------
# DepthAnything3 nn.Module
# ---------------------------------------------------------------------------

class DepthAnything3(nn.Module):
    """Depth Anything 3 depth estimation network.

    Combines a DINOv2 backbone with a DPT decoder to produce a monocular
    depth/inverse-depth map at the input image resolution.
    """

    def __init__(
        self,
        backbone: DINOv2Backbone,
        decoder: DPTDecoder,
    ):
        """Initialise DepthAnything3.

        Args:
            backbone: A ``DINOv2Backbone`` instance.
            decoder: A ``DPTDecoder`` instance.
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    @classmethod
    def from_config(
        cls,
        variant: str = "vit_l",
        img_size: int = 518,
        features: int = 256,
        patch_size: int = 14,
        pretrained: bool = True,
        num_register_tokens: int = 0,
    ) -> "DepthAnything3":
        """Convenience factory to build the model from high-level settings.

        Args:
            variant: DINOv2 size (``"vit_s"``, ``"vit_b"``, ``"vit_l"``, ``"vit_g"``).
            img_size: Square input resolution passed to the backbone and decoder.
            features: Decoder channel width.
            patch_size: ViT patch size (14 for all DINOv2 variants).
            pretrained: Load pretrained DINOv2 weights.
            num_register_tokens: Register tokens (0 for standard DINOv2).

        Returns:
            An initialised ``DepthAnything3`` model.
        """
        backbone = DINOv2Backbone(
            variant=variant,
            pretrained=pretrained,
            num_register_tokens=num_register_tokens,
        )
        embed_dim = backbone.embed_dim

        decoder = DPTDecoder(
            embed_dim=embed_dim,
            features=features,
            num_layers=len(backbone.hook_layer_ids),
            patch_size=patch_size,
            img_size=img_size,
            num_register_tokens=num_register_tokens,
            out_channels=1,
        )
        return cls(backbone=backbone, decoder=decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict an inverse-depth map.

        Args:
            x: Input image tensor, shape ``(B, 3, H, W)``.

        Returns:
            Predicted depth map, shape ``(B, 1, H, W)``.
        """
        hidden_states = self.backbone(x)
        depth = self.decoder(hidden_states)

        # Resize to input resolution if needed.
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode="bilinear", align_corners=True)

        return depth


# ---------------------------------------------------------------------------
# VisionModule wrapper
# ---------------------------------------------------------------------------

class DepthAnything3Model(VisionModule):
    """UniCV ``VisionModule`` wrapper around ``DepthAnything3``.

    Inputs:
        rgb (Modality.RGB, InputForm.SINGLE): RGB image tensor ``(B, 3, H, W)``.

    Outputs:
        Modality.DEPTH: Predicted depth / inverse-depth map.
    """

    input_spec: dict[Modality, InputForm] = {Modality.RGB: InputForm.SINGLE}
    output_modalities: list[Modality] = [Modality.DEPTH]

    def __init__(self, net: DepthAnything3):
        """Initialise DepthAnything3Model.

        Args:
            net: A pre-constructed ``DepthAnything3`` instance.
        """
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        """Run depth prediction.

        Args:
            **inputs: Must contain key ``"rgb"`` with a tensor value.

        Returns:
            ``{Modality.DEPTH: depth_tensor}``
        """
        rgb: torch.Tensor = inputs[Modality.RGB.value]
        depth = self.net(rgb)
        return {Modality.DEPTH: depth}


__all__ = [
    "DINOv2Backbone",
    "DepthAnything3",
    "DepthAnything3Model",
]

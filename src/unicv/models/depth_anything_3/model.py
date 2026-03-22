"""Depth Anything 3 depth estimation model.

Based on:
  ByteDance-Seed - Depth Anything 3 (2025)
  https://github.com/ByteDance-Seed/Depth-Anything-3

Architecture
------------
- **Backbone**: DINOv2 (ViT-L/14) loaded via ``torch.hub``.  The backbone
  exposes 4 intermediate hidden states for use by the decoder.
- **Decoder**: ``DPTDecoder`` from ``unicv.nn.dpt`` -- reassembles patch tokens
  into multi-scale spatial maps and progressively fuses them to full resolution.
- **VisionModule wrapper**: ``DepthAnything3Model`` implements the standard
  ``VisionModule`` interface (RGB-in -> DEPTH-out).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.models.base import VisionModule, _remap_checkpoint, _require_package, _warn_missing_keys
from unicv.nn.dinov2 import DINOv2Backbone
from unicv.nn.dpt import DPTDecoder
from unicv.utils.types import InputForm, Modality


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
        """Predict a depth map.

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
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        rgb: torch.Tensor = inputs[Modality.RGB.value]
        depth = self.net(rgb)
        return {Modality.DEPTH: depth}

    @classmethod
    def from_pretrained(
        cls,
        variant: str = "vit_l",
        img_size: int = 518,
        features: int = 256,
        cache_dir: str | None = None,
    ) -> "DepthAnything3Model":
        """Load official Depth Anything 3 pretrained weights.

        Downloads a model checkpoint from the ``depth-anything`` organisation on
        Hugging Face Hub and loads it into a freshly constructed
        ``DepthAnything3Model``.

        Supported variants and their Hub repos:

        =========  =====================================  ===========
        variant    repo_id                                embed_dim
        =========  =====================================  ===========
        ``vit_s``  ``depth-anything/DA3-SMALL``           384
        ``vit_b``  ``depth-anything/DA3-BASE``            768
        ``vit_l``  ``depth-anything/DA3-LARGE``  (default) 1024
        ``vit_g``  ``depth-anything/DA3-GIANT``           1536
        =========  =====================================  ===========

        Checkpoint keys are remapped from the official naming convention;
        see the source for the full mapping.  Unrecognised keys are silently
        ignored via ``strict=False``.

        Requirements: ``pip install huggingface_hub safetensors``

        Args:
            variant:   DINOv2 backbone size.  One of
                       ``"vit_s"``, ``"vit_b"``, ``"vit_l"``, ``"vit_g"``.
            img_size:  Square input resolution expected by the backbone and
                       decoder (default 518, matching official training).
            features:  Decoder channel width (default 256).
            cache_dir: Optional Hugging Face cache directory.

        Returns:
            A ``DepthAnything3Model`` with pretrained weights loaded.

        Example::

            model = DepthAnything3Model.from_pretrained("vit_l")
        """
        _require_package("huggingface_hub")
        from huggingface_hub import hf_hub_download

        _VARIANT_TO_REPO = {
            "vit_s": "depth-anything/DA3-SMALL",
            "vit_b": "depth-anything/DA3-BASE",
            "vit_l": "depth-anything/DA3-LARGE",
            "vit_g": "depth-anything/DA3-GIANT",
        }
        if variant not in _VARIANT_TO_REPO:
            raise ValueError(
                f"variant must be one of {list(_VARIANT_TO_REPO)}, got {variant!r}"
            )

        repo_id = _VARIANT_TO_REPO[variant]
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            cache_dir=cache_dir,
        )

        # Load checkpoint (safetensors preferred; fall back to torch.load).
        if ckpt_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as _load_safetensors
                raw_sd = _load_safetensors(ckpt_path)
            except ImportError as e:
                raise ImportError(
                    "Loading safetensors checkpoints requires the safetensors package.\n"
                    "Install it with:  pip install safetensors"
                ) from e
        else:
            raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Build architecture. pretrained=False because we load backbone weights
        # from the checkpoint itself (avoids a second torch.hub download).
        net = DepthAnything3.from_config(
            variant=variant,
            img_size=img_size,
            features=features,
            pretrained=False,
        )

        # Remap checkpoint keys to unicv naming.
        remapped = _remap_checkpoint(raw_sd, {"pretrained.": "backbone.model."})

        missing, _ = net.load_state_dict(remapped, strict=False)
        _warn_missing_keys(f"DepthAnything3 ({variant})", missing)

        return cls(net=net)


__all__ = [
    "DINOv2Backbone",
    "DepthAnything3",
    "DepthAnything3Model",
]

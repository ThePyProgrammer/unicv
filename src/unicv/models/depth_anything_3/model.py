"""Depth Anything 3 depth estimation model.

Based on:
  ByteDance-Seed - Depth Anything 3 (2025)
  https://github.com/ByteDance-Seed/Depth-Anything-3

Architecture
------------
- **Backbone**: DINOv2 (ViT-L/14) loaded via ``torch.hub``. The DA3
  configuration alternates per-view and cross-view attention using camera tokens.
- **Decoder**: ``DPTDecoder`` from ``unicv.nn.dpt`` -- reassembles patch tokens
  into multi-scale spatial maps and progressively fuses them to full resolution.
- **Gaussian head**: Optional image-conditioned ``GSDPT`` branch, enabled by
  default for ``vit_g``.
- **VisionModule wrapper**: ``DepthAnything3Model`` implements the standard
  ``VisionModule`` interface (RGB-in -> DEPTH, optionally SPLAT).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.models.base import VisionModule, _remap_checkpoint, _require_package, _warn_missing_keys
from unicv.nn.dinov2 import DINOv2Backbone
from unicv.nn.dpt import DPTDecoder
from unicv.nn.gsdpt import GSDPT
from unicv.utils.types import InputForm, Modality


# ---------------------------------------------------------------------------
# DepthAnything3 nn.Module
# ---------------------------------------------------------------------------

class DepthAnything3(nn.Module):
    """Depth Anything 3 depth estimation network.

    Combines a cross-view DINOv2 backbone with a DPT decoder to predict a depth
    map for each input view, with an optional Gaussian splat decoder.
    """

    def __init__(
        self,
        backbone: DINOv2Backbone,
        decoder: DPTDecoder,
        gaussian_decoder: GSDPT | None = None,
    ):
        """Initialise DepthAnything3.

        Args:
            backbone: A ``DINOv2Backbone`` instance.
            decoder: A ``DPTDecoder`` instance.
            gaussian_decoder: Optional image-conditioned GSDPT branch.
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.gaussian_decoder = gaussian_decoder

    @classmethod
    def from_config(
        cls,
        variant: str = "vit_l",
        img_size: int = 518,
        features: int = 256,
        patch_size: int = 14,
        pretrained: bool = True,
        num_register_tokens: int = 0,
        alt_start: int | None = None,
        include_gaussian_head: bool | None = None,
        sh_degree: int = 3,
    ) -> "DepthAnything3":
        """Convenience factory to build the model from high-level settings.

        Args:
            variant: DINOv2 size (``"vit_s"``, ``"vit_b"``, ``"vit_l"``, ``"vit_g"``).
            img_size: Square input resolution passed to the backbone and decoder.
            features: Decoder channel width.
            patch_size: ViT patch size (14 for all DINOv2 variants).
            pretrained: Load pretrained DINOv2 weights.
            num_register_tokens: Register tokens (0 for standard DINOv2).
            alt_start: First cross-view attention layer. Defaults to the
                transformer midpoint; pass ``-1`` to disable cross-view attention.
            include_gaussian_head: Add the GSDPT branch. Defaults to enabled
                for ``vit_g``.
            sh_degree: Spherical-harmonic degree for Gaussian colour.

        Returns:
            An initialised ``DepthAnything3`` model.
        """
        backbone = DINOv2Backbone(
            variant=variant,
            pretrained=pretrained,
            num_register_tokens=num_register_tokens,
        )
        if alt_start is None:
            alt_start = len(backbone.model.blocks) // 2
        if alt_start != -1:
            backbone.enable_cross_view_attention(alt_start)
        embed_dim = backbone.embed_dim * (
            2 if backbone.cat_token and backbone.alt_start != -1 else 1
        )

        decoder = DPTDecoder(
            embed_dim=embed_dim,
            features=features,
            num_layers=len(backbone.hook_layer_ids),
            patch_size=patch_size,
            img_size=img_size,
            num_register_tokens=num_register_tokens,
            out_channels=1,
        )
        if include_gaussian_head is None:
            include_gaussian_head = variant == "vit_g"
        gaussian_decoder = (
            GSDPT(
                embed_dim=embed_dim,
                features=features,
                num_layers=len(backbone.hook_layer_ids),
                patch_size=patch_size,
                img_size=img_size,
                num_register_tokens=num_register_tokens,
                sh_degree=sh_degree,
            )
            if include_gaussian_head
            else None
        )
        return cls(
            backbone=backbone,
            decoder=decoder,
            gaussian_decoder=gaussian_decoder,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        predict_gaussians: bool = False,
    ) -> torch.Tensor | dict[Modality, Any]:
        """Predict depth, and optionally per-view Gaussian splats.

        Args:
            x: RGB image ``(B, 3, H, W)`` or view set ``(B, S, 3, H, W)``.
            predict_gaussians: Run GSDPT and return depth plus a GaussianCloud.

        Returns:
            A depth tensor, or a modality-keyed depth/splat dictionary when
            ``predict_gaussians`` is true.
        """
        if x.ndim not in (4, 5):
            raise ValueError(
                f"Expected RGB shaped (B, 3, H, W) or (B, S, 3, H, W), got {tuple(x.shape)}"
            )
        if predict_gaussians and self.gaussian_decoder is None:
            raise ValueError("This DepthAnything3 model has no GSDPT Gaussian head")

        hidden_states = self.backbone(x)
        if x.ndim == 5:
            batch, views = x.shape[:2]
            flat_states = [
                state.reshape(batch * views, *state.shape[2:])
                for state in hidden_states
            ]
        else:
            flat_states = hidden_states
        depth = self.decoder(flat_states)

        # Resize to input resolution if needed.
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(
                depth,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        if x.ndim == 5:
            depth = depth.reshape(batch, views, *depth.shape[1:])
        if not predict_gaussians:
            return depth

        assert self.gaussian_decoder is not None
        cloud = self.gaussian_decoder(hidden_states, x)
        return {Modality.DEPTH: depth, Modality.SPLAT: cloud}


# ---------------------------------------------------------------------------
# VisionModule wrapper
# ---------------------------------------------------------------------------

class DepthAnything3Model(VisionModule):
    """UniCV ``VisionModule`` wrapper around ``DepthAnything3``.

    Inputs:
        rgb (Modality.RGB, InputForm.SINGLE): RGB image ``(B, 3, H, W)`` or
            multi-view tensor ``(B, S, 3, H, W)``.

    Outputs:
        Modality.DEPTH: Predicted per-view depth map(s).
        Modality.SPLAT: Per-view GaussianCloud, when GSDPT is configured.
    """

    input_spec: dict[Modality, InputForm] = {Modality.RGB: InputForm.SINGLE}
    output_modalities: list[Modality] = [Modality.DEPTH, Modality.SPLAT]

    def __init__(self, net: DepthAnything3):
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        rgb: torch.Tensor = inputs[Modality.RGB.value]
        prediction = self.net(
            rgb,
            predict_gaussians=self.net.gaussian_decoder is not None,
        )
        if isinstance(prediction, dict):
            return prediction
        return {Modality.DEPTH: prediction}

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

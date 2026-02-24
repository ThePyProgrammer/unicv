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

        Key mapping applied before loading
        -----------------------------------
        The official checkpoints use a different module naming convention:

        * ``pretrained.*``                      → ``backbone.model.*``
        * ``depth_head.projects.{i}.*``         → ``decoder.reassemble_blocks.{i}.project.*``
        * ``depth_head.resize_layers.{i}.*``    → ``decoder.reassemble_blocks.{i}.resample.*``
        * ``depth_head.scratch.refinenet{n}.*`` → ``decoder.fusion_blocks.{4-n}.*``
          (with ``resConfUnit`` renamed to ``resConvUnit``)
        * ``depth_head.scratch.output_conv1.*`` → ``decoder.head.0.*``
        * ``depth_head.scratch.output_conv2.*`` → ``decoder.head.2.*``

        ``strict=False`` is used so that keys present in the checkpoint but
        absent from the unicv architecture (e.g. ``depth_head.scratch.layer_rn*``
        intermediate convolutions) are silently ignored.

        Requirements
        ------------
        ``pip install huggingface_hub safetensors``

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
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "DepthAnything3Model.from_pretrained requires huggingface_hub.\n"
                "Install it with:  pip install huggingface_hub"
            )

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
            except ImportError:
                raise ImportError(
                    "Loading safetensors checkpoints requires the safetensors package.\n"
                    "Install it with:  pip install safetensors"
                )
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
        remapped: dict[str, torch.Tensor] = {}
        for key, val in raw_sd.items():
            new_key = key

            if new_key.startswith("pretrained."):
                # DINOv2 backbone weights.
                new_key = "backbone.model." + new_key[len("pretrained."):]

            elif new_key.startswith("depth_head.projects."):
                # Reassemble project conv: depth_head.projects.{i}.* → reassemble_blocks.{i}.project.*
                rest = new_key[len("depth_head.projects."):]
                idx, _, tail = rest.partition(".")
                new_key = f"decoder.reassemble_blocks.{idx}.project.{tail}"

            elif new_key.startswith("depth_head.resize_layers."):
                # Reassemble spatial resample: depth_head.resize_layers.{i}.* → reassemble_blocks.{i}.resample.*
                rest = new_key[len("depth_head.resize_layers."):]
                idx, _, tail = rest.partition(".")
                new_key = f"decoder.reassemble_blocks.{idx}.resample.{tail}"

            elif new_key.startswith("depth_head.scratch.refinenet"):
                # Fusion blocks — stored in reverse order in the checkpoint.
                # refinenet4 (coarsest) → fusion_blocks[3]; refinenet1 (finest) → fusion_blocks[0].
                rest = new_key[len("depth_head.scratch.refinenet"):]
                n_str, _, tail = rest.partition(".")
                i = 4 - int(n_str)            # refinenet4→0, refinenet3→1, …
                tail = tail.replace("resConfUnit", "resConvUnit")
                new_key = f"decoder.fusion_blocks.{i}.{tail}"

            elif new_key.startswith("depth_head.scratch.output_conv1."):
                tail = new_key[len("depth_head.scratch.output_conv1."):]
                new_key = f"decoder.head.0.{tail}"

            elif new_key.startswith("depth_head.scratch.output_conv2."):
                tail = new_key[len("depth_head.scratch.output_conv2."):]
                new_key = f"decoder.head.2.{tail}"

            # Keys that don't match any pattern (e.g. depth_head.scratch.layer_rn*)
            # are kept as-is and will fall through strict=False unmatched.
            remapped[new_key] = val

        missing, _ = net.load_state_dict(remapped, strict=False)
        if missing:
            import warnings
            shown = missing[:5]
            warnings.warn(
                f"DepthAnything3 ({variant}): {len(missing)} missing key(s) when "
                f"loading pretrained weights (first 5): {shown}",
                stacklevel=2,
            )

        return cls(net=net)


__all__ = [
    "DINOv2Backbone",
    "DepthAnything3",
    "DepthAnything3Model",
]

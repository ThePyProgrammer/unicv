"""Camera Depth Model (CDM).

A dual-ViT depth estimator: one ViT branch processes RGB, another processes
the raw (noisy) depth signal.  At each of four intermediate encoder levels
the two token streams are fused via an additive learned projection, and the
fused states are decoded by a DPTDecoder into a refined depth map.

Architecture reference
-----------------------
    Manipulation as in Simulation – Camera Depth Model
    https://manipulation-as-in-simulation.github.io/#cdm-results

This implementation follows the documented design:
- ``TokenFusion``     – per-level additive fusion: fused = rgb + proj(depth)
- ``CDM``             – wraps two backbones (any callable returning
                         ``List[Tensor(B, N, D)]``), N fusion layers,
                         and a ``DPTDecoder``
- ``CameraDepthModel``– ``VisionModule`` wrapper

VisionModule spec
-----------------
    Inputs:  Modality.RGB   (SINGLE): ``(B, 3, H, W)``
             Modality.DEPTH (SINGLE): ``(B, 1, H, W)`` raw/noisy depth
    Outputs: Modality.DEPTH         : refined depth map ``(B, 1, H, W)``
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.models.base import VisionModule, _remap_dpt_key, _require_package, _warn_missing_keys
from unicv.nn.dpt import DPTDecoder
from unicv.utils.types import InputForm, Modality


# ---------------------------------------------------------------------------
# Token fusion
# ---------------------------------------------------------------------------

class TokenFusion(nn.Module):
    """Fuse RGB and depth patch-token streams at one encoder level.

    Applies a learned linear projection to the depth tokens then adds them
    to the RGB tokens (residual addition keeps gradient flow stable).

    Args:
        embed_dim: Token feature dimension ``D``.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused tokens.

        Args:
            rgb_tokens:   Shape ``(B, N, D)``.
            depth_tokens: Shape ``(B, N, D)``.

        Returns:
            Fused tokens, shape ``(B, N, D)``.
        """
        return rgb_tokens + self.proj(depth_tokens)


# ---------------------------------------------------------------------------
# CDM nn.Module
# ---------------------------------------------------------------------------

class CDM(nn.Module):
    """Camera Depth Model – dual-ViT encoder with per-level token fusion.

    Both ``rgb_backbone`` and ``depth_backbone`` must be callable modules
    that accept a ``(B, 3, H, W)`` tensor and return a list of
    ``num_levels`` hidden-state tensors each of shape ``(B, 1+extra+N, D)``
    (i.e. the DINOv2Backbone / any hooked ViT interface).

    A single-channel depth map is first projected to 3 channels before being
    passed to ``depth_backbone``.

    Args:
        rgb_backbone:   ViT backbone for the RGB stream.
        depth_backbone: ViT backbone for the depth stream.
        decoder:        DPT decoder that consumes fused hidden states.
        embed_dim:      Hidden-state dimensionality ``D`` of both backbones.
        num_levels:     Number of hooked encoder levels (default 4).
    """

    def __init__(
        self,
        rgb_backbone: nn.Module,
        depth_backbone: nn.Module,
        decoder: DPTDecoder,
        embed_dim: int,
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        self.rgb_backbone   = rgb_backbone
        self.depth_backbone = depth_backbone
        self.decoder        = decoder

        # Lift single-channel depth to 3 channels for the depth backbone.
        self.depth_proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        # One fusion layer per encoder level.
        self.fusion_layers = nn.ModuleList(
            [TokenFusion(embed_dim) for _ in range(num_levels)]
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a refined depth map.

        Args:
            rgb:   RGB image, shape ``(B, 3, H, W)``.
            depth: Raw depth map, shape ``(B, 1, H, W)``.

        Returns:
            Refined depth map, shape ``(B, 1, H, W)`` (upsampled to input
            resolution when the decoder produces a coarser output).
        """
        depth_3ch: torch.Tensor = self.depth_proj(depth)   # (B, 3, H, W)

        rgb_states:   list[torch.Tensor] = self.rgb_backbone(rgb)
        depth_states: list[torch.Tensor] = self.depth_backbone(depth_3ch)

        fused = [
            fusion(r, d)
            for fusion, r, d in zip(self.fusion_layers, rgb_states, depth_states)
        ]

        out: torch.Tensor = self.decoder(fused)             # (B, 1, H', W')

        if out.shape[-2:] != rgb.shape[-2:]:
            out = F.interpolate(
                out, size=rgb.shape[-2:], mode="bilinear", align_corners=True
            )
        return out

    @classmethod
    def from_config(
        cls,
        variant: str = "vit_l",
        img_size: int = 518,
        features: int = 256,
        patch_size: int = 14,
        pretrained: bool = True,
        num_register_tokens: int = 0,
    ) -> "CDM":
        """Build a CDM from high-level configuration.

        Both backbone branches use the same DINOv2 variant but have
        independent weights (they can diverge during fine-tuning).

        Args:
            variant:             DINOv2 variant (``"vit_s"``/``"vit_b"``/
                                 ``"vit_l"``/``"vit_g"``).
            img_size:            Input resolution (square).
            features:            Decoder channel width.
            patch_size:          ViT patch size (14 for all DINOv2 variants).
            pretrained:          Whether to initialise from DINOv2 weights.
            num_register_tokens: Register tokens (0 for standard DINOv2).

        Returns:
            Initialised ``CDM`` model.
        """
        from unicv.nn.dinov2 import DINOv2Backbone

        rgb_backbone   = DINOv2Backbone(variant, pretrained, num_register_tokens=num_register_tokens)
        depth_backbone = DINOv2Backbone(variant, pretrained, num_register_tokens=num_register_tokens)
        embed_dim      = rgb_backbone.embed_dim
        num_levels     = len(rgb_backbone.hook_layer_ids)

        decoder = DPTDecoder(
            embed_dim=embed_dim,
            features=features,
            num_layers=num_levels,
            patch_size=patch_size,
            img_size=img_size,
            num_register_tokens=num_register_tokens,
            out_channels=1,
        )
        return cls(
            rgb_backbone=rgb_backbone,
            depth_backbone=depth_backbone,
            decoder=decoder,
            embed_dim=embed_dim,
            num_levels=num_levels,
        )


# ---------------------------------------------------------------------------
# VisionModule wrapper
# ---------------------------------------------------------------------------

class CameraDepthModel(VisionModule):
    """UniCV ``VisionModule`` wrapper around ``CDM``.

    Inputs:
        rgb   (Modality.RGB,   InputForm.SINGLE): ``(B, 3, H, W)``
        depth (Modality.DEPTH, InputForm.SINGLE): ``(B, 1, H, W)`` raw depth

    Outputs:
        Modality.DEPTH: Refined depth map ``(B, 1, H, W)``.
    """

    input_spec: dict[Modality, InputForm] = {
        Modality.RGB:   InputForm.SINGLE,
        Modality.DEPTH: InputForm.SINGLE,
    }
    output_modalities: list[Modality] = [Modality.DEPTH]

    def __init__(self, net: CDM) -> None:
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        rgb:   torch.Tensor = inputs[Modality.RGB.value]
        depth: torch.Tensor = inputs[Modality.DEPTH.value]
        out = self.net(rgb, depth)
        return {Modality.DEPTH: out}

    @classmethod
    def from_pretrained(
        cls,
        camera: str = "d405",
        img_size: int = 518,
        features: int = 256,
        cache_dir: str | None = None,
    ) -> "CameraDepthModel":
        """Load official Camera Depth Model pretrained weights.

        Downloads a camera-specific checkpoint from the ``depth-anything``
        organisation on Hugging Face Hub and loads it into a freshly
        constructed ``CameraDepthModel``.

        The CDM checkpoints are trained per depth-camera sensor because each
        sensor has a distinct noise distribution.  Choose ``camera`` to match
        your hardware:

        ============  =====================================================
        camera        Hugging Face repo
        ============  =====================================================
        ``"d405"``    ``depth-anything/camera-depth-model-d405``  (default)
        ``"d435"``    ``depth-anything/camera-depth-model-d435``
        ``"l515"``    ``depth-anything/camera-depth-model-l515``
        ``"kinect"``  ``depth-anything/camera-depth-model-kinect``
        ============  =====================================================

        Checkpoint keys are remapped from the official naming convention;
        see the source for the full mapping.  Unrecognised keys are silently
        ignored via ``strict=False``.

        Requirements: ``pip install huggingface_hub``

        Args:
            camera:    Depth-camera variant.  One of
                       ``"d405"``, ``"d435"``, ``"l515"``, ``"kinect"``.
            img_size:  Square input resolution (default 518).
            features:  Decoder channel width (default 256).
            cache_dir: Optional Hugging Face cache directory.

        Returns:
            A ``CameraDepthModel`` with pretrained weights loaded.

        Example::

            model = CameraDepthModel.from_pretrained("d405")
        """
        _require_package("huggingface_hub")
        from huggingface_hub import hf_hub_download

        _VALID_CAMERAS = {"d405", "d435", "l515", "kinect"}
        if camera not in _VALID_CAMERAS:
            raise ValueError(
                f"camera must be one of {sorted(_VALID_CAMERAS)}, got {camera!r}"
            )

        repo_id = f"depth-anything/camera-depth-model-{camera}"
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.pth",
            cache_dir=cache_dir,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Build CDM with ViT-L (matching the official training config) but
        # without loading DINOv2 pretrained weights from torch.hub — backbone
        # weights come from the CDM checkpoint itself.
        net = CDM.from_config(
            variant="vit_l",
            img_size=img_size,
            features=features,
            pretrained=False,
        )

        # Remap checkpoint keys to unicv CDM naming.
        remapped: dict[str, torch.Tensor] = {}
        for key, val in state_dict.items():
            new_key = key

            # --- RGB backbone ---
            if new_key.startswith("pretrained.") or new_key.startswith("rgb_encoder."):
                prefix_len = len("pretrained.") if new_key.startswith("pretrained.") else len("rgb_encoder.")
                new_key = "rgb_backbone.model." + new_key[prefix_len:]

            # --- Depth backbone ---
            elif new_key.startswith("depth_encoder."):
                new_key = "depth_backbone.model." + new_key[len("depth_encoder."):]

            else:
                dpt_key = _remap_dpt_key(new_key)
                if dpt_key is not None:
                    new_key = dpt_key

            # depth_proj.*, fusion_layers.*, decoder.* — pass through unchanged.
            remapped[new_key] = val

        missing, _ = net.load_state_dict(remapped, strict=False)
        _warn_missing_keys(f"CameraDepthModel ({camera})", missing)

        return cls(net=net)


__all__ = ["TokenFusion", "CDM", "CameraDepthModel"]

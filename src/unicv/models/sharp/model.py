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


    @classmethod
    def from_pretrained(
        cls,
        sh_degree: int = 0,
        features: int = 256,
        img_size: int = 518,
        cache_dir: str | None = None,
    ) -> "SHARPModel":
        """Load the official Apple SHARP pretrained weights.

        Downloads ``sharp_2572gikvuh.pt`` from Apple's CDN and loads it into a
        freshly constructed ``SHARPModel``.

        The official checkpoint stores a ``RGBGaussianPredictor`` whose encoder
        is DepthPro-based (``SlidingPyramidNetwork`` + ``TimmViT``), not DINOv2.
        The following partial remappings are applied where shapes align:

        * ``gaussian_decoder.fusions.{i}.resnet1.residual.1.*``
          → ``feature_decoder.fusion_blocks.{i}.resConvUnit1.conv1.*``
        * ``gaussian_decoder.fusions.{i}.resnet1.residual.3.*``
          → ``feature_decoder.fusion_blocks.{i}.resConvUnit1.conv2.*``
        * ``gaussian_decoder.fusions.{i}.resnet2.residual.1.*``
          → ``feature_decoder.fusion_blocks.{i}.resConvUnit2.conv1.*``
        * ``gaussian_decoder.fusions.{i}.resnet2.residual.3.*``
          → ``feature_decoder.fusion_blocks.{i}.resConvUnit2.conv2.*``
        * ``gaussian_decoder.fusions.{i}.out_conv.*``
          → ``feature_decoder.fusion_blocks.{i}.out_conv.*``
        * ``prediction_head.geometry_prediction_head.*``
          → ``gaussian_head.xyz_head.*``  *(shape-filtered)*

        Any remapped key whose tensor shape does not match the model parameter
        is silently dropped before ``load_state_dict`` to prevent
        ``RuntimeError`` on shape mismatches.  ``strict=False`` is used so
        that unrecognised checkpoint keys are ignored.  A ``UserWarning`` is
        issued listing the first few missing keys.

        Args:
            sh_degree:  Spherical-harmonic degree for the colour head (0–3).
            features:   Decoder channel width (default 256).
            img_size:   Square input resolution passed to the decoder
                        (default 518).
            cache_dir:  Directory for the downloaded checkpoint.  Defaults to
                        the PyTorch hub cache
                        (``~/.cache/torch/hub/checkpoints``).

        Returns:
            A ``SHARPModel`` with available pretrained weights loaded.

        Example::

            model = SHARPModel.from_pretrained()
        """
        _CKPT_URL = (
            "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
        )

        state_dict = torch.hub.load_state_dict_from_url(
            _CKPT_URL,
            model_dir=cache_dir,
            map_location="cpu",
            weights_only=True,
        )

        # Build a fresh model. pretrained=False because the official SHARP
        # uses a DepthPro encoder, not DINOv2; backbone weights are not
        # transferable from this checkpoint.
        net = SHARP.from_config(
            variant="vit_l",
            img_size=img_size,
            features=features,
            sh_degree=sh_degree,
            pretrained=False,
        )

        # Remap official RGBGaussianPredictor keys → unicv SHARP keys.
        remapped: dict[str, torch.Tensor] = {}
        for key, val in state_dict.items():
            new_key: str | None = None

            # gaussian_decoder.fusions.{i}.resnet{1,2}.residual.{1,3}.*
            #   → feature_decoder.fusion_blocks.{i}.resConvUnit{1,2}.conv{1,2}.*
            if key.startswith("gaussian_decoder.fusions."):
                rest         = key[len("gaussian_decoder.fusions."):]
                idx, _, tail = rest.partition(".")
                if tail.startswith("resnet1.residual.1."):
                    new_key = (
                        f"feature_decoder.fusion_blocks.{idx}"
                        f".resConvUnit1.conv1.{tail[len('resnet1.residual.1.'):]}"
                    )
                elif tail.startswith("resnet1.residual.3."):
                    new_key = (
                        f"feature_decoder.fusion_blocks.{idx}"
                        f".resConvUnit1.conv2.{tail[len('resnet1.residual.3.'):]}"
                    )
                elif tail.startswith("resnet2.residual.1."):
                    new_key = (
                        f"feature_decoder.fusion_blocks.{idx}"
                        f".resConvUnit2.conv1.{tail[len('resnet2.residual.1.'):]}"
                    )
                elif tail.startswith("resnet2.residual.3."):
                    new_key = (
                        f"feature_decoder.fusion_blocks.{idx}"
                        f".resConvUnit2.conv2.{tail[len('resnet2.residual.3.'):]}"
                    )
                elif tail.startswith("out_conv."):
                    new_key = (
                        f"feature_decoder.fusion_blocks.{idx}"
                        f".out_conv.{tail[len('out_conv.'):]}"
                    )

            # prediction_head.geometry_prediction_head.*
            #   → gaussian_head.xyz_head.*
            # (the official head may output 3×num_layers channels rather than
            # 3; the shape filter below will drop it when that is the case.)
            elif key.startswith("prediction_head.geometry_prediction_head."):
                new_key = (
                    "gaussian_head.xyz_head."
                    + key[len("prediction_head.geometry_prediction_head."):]
                )

            if new_key is not None:
                remapped[new_key] = val

        # Drop remapped keys whose shape does not match the model parameter
        # to prevent RuntimeError from load_state_dict.
        current_sd = net.state_dict()
        safe = {
            k: v for k, v in remapped.items()
            if k in current_sd and current_sd[k].shape == v.shape
        }

        missing, _ = net.load_state_dict(safe, strict=False)
        if missing:
            import warnings
            shown = missing[:5]
            warnings.warn(
                f"SHARPModel: {len(missing)} missing key(s) when loading "
                f"pretrained weights (first 5): {shown}. "
                "Note: the official SHARP checkpoint uses a DepthPro-based "
                "encoder rather than DINOv2; backbone and reassemble-block "
                "weights are not transferable without an architectural "
                "realignment with the official RGBGaussianPredictor.",
                stacklevel=2,
            )

        return cls(net=net)


__all__ = ["SHARP", "SHARPModel"]

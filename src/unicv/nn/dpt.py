"""DPT (Dense Prediction Transformer) decoder architecture.

Implements the decoder architecture from:
  Ranftl et al. - Vision Transformers for Dense Prediction (2021)
  https://arxiv.org/abs/2103.13413
  https://github.com/isl-org/DPT

The DPT decoder reassembles patch tokens extracted from intermediate layers
of a ViT backbone into spatial feature maps at multiple resolutions, then
progressively fuses them into a single full-resolution prediction map via
FeatureFusionBlocks.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual building block
# ---------------------------------------------------------------------------

class ResidualConvUnit(nn.Module):
    """Two-layer residual conv block used inside FeatureFusionBlock."""

    def __init__(self, features: int, use_bn: bool = False):
        """Initialise ResidualConvUnit.

        Args:
            features: Number of input/output channels.
            use_bn: Whether to use BatchNorm after each conv.
        """
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        out = self.relu(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        return out + x


# ---------------------------------------------------------------------------
# Feature-fusion block
# ---------------------------------------------------------------------------

class FeatureFusionBlock(nn.Module):
    """Fusion block that merges two feature maps and upsamples by 2×.

    Used to progressively build up the full-resolution prediction from the
    coarsest encoder level to the finest.
    """

    def __init__(
        self,
        features: int,
        use_bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
    ):
        """Initialise FeatureFusionBlock.

        Args:
            features: Channel dimension of input and output feature maps.
            use_bn: Whether to add BatchNorm inside residual units.
            expand: If True, halve the output channels (useful for compact
                decoders where each level is half the width of the previous).
            align_corners: Passed to ``F.interpolate`` for bilinear upsampling.
        """
        super().__init__()
        self.align_corners = align_corners
        out_features = features // 2 if expand else features

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConvUnit1 = ResidualConvUnit(features, use_bn)
        self.resConvUnit2 = ResidualConvUnit(features, use_bn)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Fuse one or two feature maps and upsample.

        Args:
            *xs: One or two tensors. If two are given, ``xs[1]`` is processed
                through the first residual unit and added to ``xs[0]`` before
                the second residual unit.

        Returns:
            Upsampled fused feature map.
        """
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConvUnit1(xs[1])
        output = self.resConvUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


# ---------------------------------------------------------------------------
# Reassemble blocks: convert ViT tokens → spatial feature maps
# ---------------------------------------------------------------------------

class Reassemble(nn.Module):
    """Convert flat ViT patch tokens into a 2-D feature map.

    Optionally projects the channel dimension and re-samples to the desired
    spatial resolution via a convolutional ``resample`` operation.
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        patch_size: int,
        img_size: int,
        layer_scale: int,
        num_register_tokens: int = 0,
    ):
        """Initialise Reassemble.

        Args:
            embed_dim: ViT hidden dimension.
            out_channels: Output channels after projection.
            patch_size: ViT patch size (e.g. 16 or 14).
            img_size: Input image resolution (assumed square).
            layer_scale: Spatial scale relative to the patch grid.  A value
                of 1 means keep the native resolution; 2 doubles the spatial
                size via a ConvTranspose; 0.5 halves it via Conv with stride 2.
            num_register_tokens: Extra register tokens prepended to the patch
                tokens (DinoV2 uses 4).
        """
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.num_register_tokens = num_register_tokens

        # 1. Project from embed_dim → out_channels.
        self.project = nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=True)

        # 2. Spatial re-sampling.
        if layer_scale == 1:
            self.resample = nn.Identity()
        elif layer_scale == 2:
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2, padding=0
            )
        elif layer_scale == 4:
            self.resample = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
            )
        elif layer_scale == 0.5:
            self.resample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported layer_scale={layer_scale}.")

        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens to a spatial feature map.

        Args:
            tokens: Shape ``(B, 1 + num_register_tokens + N, D)`` where
                ``N = (img_size // patch_size)^2``.

        Returns:
            Spatial feature map, shape ``(B, out_channels, H', W')``.
        """
        B = tokens.shape[0]
        # Drop [CLS] token and register tokens; keep only patch tokens.
        tokens = tokens[:, 1 + self.num_register_tokens:, :]  # (B, N, D)

        h = w = self.img_size // self.patch_size
        # Reshape to 2-D spatial.
        x = tokens.permute(0, 2, 1).reshape(B, -1, h, w)  # (B, D, h, w)

        x = self.project(x)
        x = self.resample(x)
        return x


# ---------------------------------------------------------------------------
# Full DPT Decoder
# ---------------------------------------------------------------------------

class DPTDecoder(nn.Module):
    """DPT decoder for dense prediction from a ViT backbone.

    The decoder:
    1. Receives intermediate hidden states from the backbone encoder.
    2. Reassembles them into spatial multi-scale feature maps.
    3. Fuses them progressively from coarse to fine via FeatureFusionBlocks.
    4. Applies a final output projection.
    """

    DEFAULT_LAYER_SCALES = [4, 2, 1, 0.5]

    def __init__(
        self,
        embed_dim: int,
        features: int,
        num_layers: int = 4,
        patch_size: int = 16,
        img_size: int = 518,
        layer_scales: Optional[List[int]] = None,
        use_bn: bool = False,
        num_register_tokens: int = 0,
        out_channels: int = 1,
    ):
        """Initialise DPTDecoder.

        Args:
            embed_dim: ViT hidden dimension.
            features: Width (# channels) inside the decoder.
            num_layers: Number of encoder layers to hook / reassemble (typically 4).
            patch_size: ViT patch size.
            img_size: Backbone input image size (assumed square).
            layer_scales: Spatial scale  for each reassembled level.
                Defaults to ``[4, 2, 1, 0.5]`` (from finest to coarsest).
            use_bn: Whether to use BatchNorm in fusion blocks.
            num_register_tokens: Extra register tokens (e.g. 4 for DINOv2).
            out_channels: Final output channels (e.g. 1 for depth, 256 for features).
        """
        super().__init__()
        if layer_scales is None:
            layer_scales = self.DEFAULT_LAYER_SCALES[:num_layers]

        assert len(layer_scales) == num_layers, (
            f"Expected {num_layers} layer_scales, got {len(layer_scales)}"
        )

        # Reassemble blocks: one per hooked encoder layer.
        self.reassemble_blocks = nn.ModuleList([
            Reassemble(
                embed_dim=embed_dim,
                out_channels=features,
                patch_size=patch_size,
                img_size=img_size,
                layer_scale=ls,
                num_register_tokens=num_register_tokens,
            )
            for ls in layer_scales
        ])

        # Fusion blocks: fuse from coarsest (last) to finest (index 0).
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(features=features, use_bn=use_bn)
            for _ in range(num_layers)
        ])

        # Final head: project to output channels.
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale hidden states into a dense prediction.

        Args:
            hidden_states: List of ``num_layers`` tensors, each of shape
                ``(B, 1 + num_register_tokens + N, D)``, ordered from
                the shallowest to the deepest encoder layer.

        Returns:
            Dense prediction map; shape depends on head configuration.
        """
        assert len(hidden_states) == len(self.reassemble_blocks), (
            f"Expected {len(self.reassemble_blocks)} hidden states, "
            f"got {len(hidden_states)}"
        )

        # Reassemble each level into a spatial feature map.
        features = [rb(h) for rb, h in zip(self.reassemble_blocks, hidden_states)]

        # Fuse from coarsest to finest.
        x = self.fusion_blocks[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            x = self.fusion_blocks[i](x, features[i])

        return self.head(x)


__all__ = [
    "ResidualConvUnit",
    "FeatureFusionBlock",
    "Reassemble",
    "DPTDecoder",
]

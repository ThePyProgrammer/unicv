"""Multi-resolution Convolutional Decoder for DepthPro.

Ported and adapted from apple/ml-depth-pro:
  https://github.com/apple/ml-depth-pro/blob/main/src/depth_pro/network/decoder.py

Implements a variant of Vision Transformers for Dense Prediction:
  https://arxiv.org/abs/2103.13413
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Generic residual block.

    Implements residual connections from:
        He et al. - Identity Mappings in Deep Residual Networks (2016),
        https://arxiv.org/abs/1603.05027
    """

    def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
        """Initialize ResidualBlock."""
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        delta_x = self.residual(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + delta_x


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion block for the DPT-style multires decoder."""

    def __init__(
        self,
        num_features: int,
        deconv: bool = False,
        batch_norm: bool = False,
    ):
        """Initialize FeatureFusionBlock2d.

        Args:
            num_features: Input and output channel dimensions.
            deconv: If True, apply a transposed convolution (2× upsample) before
                the output convolution.
            batch_norm: If True, add BatchNorm inside residual blocks.
        """
        super().__init__()

        self.resnet1 = self._residual_block(num_features, batch_norm)
        self.resnet2 = self._residual_block(num_features, batch_norm)

        self.use_deconv = deconv
        if deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.out_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
        """Process and fuse input feature maps.

        Args:
            x0: Primary feature map (lower resolution, from previous fusion level).
            x1: Optional skip feature map to add before the second residual block.

        Returns:
            Fused and (optionally) upsampled feature map.
        """
        x = x0
        if x1 is not None:
            res = self.resnet1(x1)
            x = self.skip_add.add(x, res)

        x = self.resnet2(x)

        if self.use_deconv:
            x = self.deconv(x)
        x = self.out_conv(x)
        return x

    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool) -> ResidualBlock:
        """Create a two-layer residual block."""

        def _block(dim: int, bn: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.ReLU(False),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=not bn),
            ]
            if bn:
                layers.append(nn.BatchNorm2d(dim))
            return layers

        residual = nn.Sequential(
            *_block(num_features, batch_norm),
            *_block(num_features, batch_norm),
        )
        return ResidualBlock(residual)


class MultiresConvDecoder(nn.Module):
    """Multi-resolution convolutional decoder.

    Fuses multi-scale encoder feature maps (from coarsest to finest) into a
    single high-resolution feature map, ready for the prediction head.
    """

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dim_decoder: int,
    ):
        """Initialize MultiresConvDecoder.

        Args:
            dims_encoder: Channel dimensions at each encoder level (finest first).
            dim_decoder: Uniform channel dimension inside the decoder.
        """
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder
        self.dim_out = dim_decoder

        num_encoders = len(self.dims_encoder)

        # At the highest resolution (level 0) use a 1×1 conv when dims differ.
        conv0 = (
            nn.Conv2d(self.dims_encoder[0], dim_decoder, kernel_size=1, bias=False)
            if self.dims_encoder[0] != dim_decoder
            else nn.Identity()
        )

        convs: list[nn.Module] = [conv0]
        for i in range(1, num_encoders):
            convs.append(
                nn.Conv2d(
                    self.dims_encoder[i],
                    dim_decoder,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
        self.convs = nn.ModuleList(convs)

        fusions: list[nn.Module] = []
        for i in range(num_encoders):
            fusions.append(
                FeatureFusionBlock2d(
                    num_features=dim_decoder,
                    deconv=(i != 0),
                    batch_norm=False,
                )
            )
        self.fusions = nn.ModuleList(fusions)

    def forward(
        self, encodings: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a list of multi-resolution feature maps.

        Args:
            encodings: Feature maps ordered finest → coarsest, each of shape
                ``(B, C_i, H_i, W_i)``.

        Returns:
            A tuple ``(features, lowres_features)`` where *features* is the
            high-resolution fused output and *lowres_features* is the projected
            coarsest feature map (used e.g. for FOV estimation).
        """
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)

        if num_levels != num_encoders:
            raise ValueError(
                f"Got encoder output levels={num_levels}, expected levels={num_encoders}."
            )

        # Project + fuse from coarsest (last) to finest (first).
        features = self.convs[-1](encodings[-1])
        lowres_features = features
        features = self.fusions[-1](features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)
        return features, lowres_features


__all__ = [
    "ResidualBlock",
    "FeatureFusionBlock2d",
    "MultiresConvDecoder",
]

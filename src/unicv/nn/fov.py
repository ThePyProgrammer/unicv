"""Field-of-View estimation network for DepthPro.

Ported from apple/ml-depth-pro:
  https://github.com/apple/ml-depth-pro/blob/main/src/depth_pro/network/fov.py
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FOVNetwork(nn.Module):
    """Field of View estimation network.

    Given an input image and a low-resolution feature map produced by the
    encoder, estimates a scalar field-of-view (in degrees) per image in the
    batch.  An optional ViT-based ``fov_encoder`` can be provided to bring
    extra capacity; otherwise the low-resolution encoder features are used
    directly.
    """

    def __init__(
        self,
        num_features: int,
        fov_encoder: Optional[nn.Module] = None,
    ):
        """Initialize FOVNetwork.

        Args:
            num_features: Feature dimension ``D`` from the decoder.
            fov_encoder: Optional ViT backbone used to encode a downsampled
                copy of the input image. Its ``embed_dim`` must be compatible
                with ``num_features``.
        """
        super().__init__()

        # Build the convolutional head in two parts so that when an encoder is
        # present the downsampling step is separated from the rest.
        fov_head0: list[nn.Module] = [
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        ]
        fov_head: list[nn.Module] = [
            nn.Conv2d(num_features // 2, num_features // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1, padding=0),
        ]

        if fov_encoder is not None:
            self.encoder = nn.Sequential(
                fov_encoder,
                nn.Linear(fov_encoder.embed_dim, num_features // 2),
            )
            self.downsample = nn.Sequential(*fov_head0)
        else:
            fov_head = fov_head0 + fov_head

        self.head = nn.Sequential(*fov_head)

    def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
        """Estimate field of view.

        Args:
            x: Input RGB image, shape ``(B, 3, H, W)``.
            lowres_feature: Low-resolution encoder feature map,
                shape ``(B, D, h, w)``.

        Returns:
            Scalar FOV prediction per image, shape ``(B, 1)``.
        """
        if hasattr(self, "encoder"):
            # Downsample the image to backbone resolution and encode.
            x = F.interpolate(
                x,
                size=None,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            )
            x = self.encoder(x)[:, 1:].permute(0, 2, 1)
            lowres_feature = self.downsample(lowres_feature)
            x = x.reshape_as(lowres_feature) + lowres_feature
        else:
            x = lowres_feature
        return self.head(x)


__all__ = ["FOVNetwork"]

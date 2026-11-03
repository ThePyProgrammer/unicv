"""Image-conditioned DPT head for per-pixel Gaussian splat prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.nn.dpt import DPTDecoder
from unicv.nn.gaussian import GaussianHead
from unicv.utils.structs import GaussianCloud


class GSDPT(nn.Module):
    """DPT feature decoder with an image-conditioned Gaussian prediction head.

    The DPT feature map is merged with a shallow RGB branch before predicting
    one Gaussian per output pixel. Inputs may contain one image per batch item
    or a batch of multi-view image sets.
    """

    def __init__(
        self,
        embed_dim: int,
        features: int,
        num_layers: int = 4,
        patch_size: int = 14,
        img_size: int = 518,
        num_register_tokens: int = 0,
        sh_degree: int = 3,
    ) -> None:
        super().__init__()
        if features < 8:
            raise ValueError(f"features must be at least 8, got {features}")

        self.decoder = DPTDecoder(
            embed_dim=embed_dim,
            features=features,
            num_layers=num_layers,
            patch_size=patch_size,
            img_size=img_size,
            num_register_tokens=num_register_tokens,
            out_channels=features,
        )
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, features // 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 8, features // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 4, features, 3, padding=1),
            nn.GELU(),
        )
        self.gaussian_head = GaussianHead(features, sh_degree=sh_degree)

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        images: torch.Tensor,
    ) -> GaussianCloud:
        """Predict Gaussian parameters from intermediate features and RGB.

        Args:
            hidden_states: DINO states shaped ``(B, N, D)`` or
                ``(B, S, N, D)``.
            images: RGB images shaped ``(B, 3, H, W)`` or
                ``(B, S, 3, H, W)``.
        """
        if not hidden_states:
            raise ValueError("GSDPT requires at least one hidden state")
        if images.ndim == 4:
            batch, views = images.shape[0], 1
            image_batch = images
            has_view_axis = False
        elif images.ndim == 5:
            batch, views = images.shape[:2]
            image_batch = images.reshape(batch * views, *images.shape[2:])
            has_view_axis = True
        else:
            raise ValueError(
                f"Expected images shaped (B, 3, H, W) or (B, S, 3, H, W), got {tuple(images.shape)}"
            )
        if image_batch.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB images, got {image_batch.shape[1]} channels")

        if hidden_states[0].ndim == 4:
            if hidden_states[0].shape[:2] != (batch, views):
                raise ValueError("Hidden-state batch and view dimensions must match images")
            states = [state.reshape(batch * views, *state.shape[2:]) for state in hidden_states]
        elif hidden_states[0].ndim == 3:
            states = hidden_states
            if states[0].shape[0] != batch * views:
                raise ValueError("Hidden-state batch dimension must match images")
        else:
            raise ValueError("Hidden states must have 3 or 4 dimensions")

        feature_map = self.decoder.head(self.decoder.forward_features(states))
        if feature_map.shape[-2:] != image_batch.shape[-2:]:
            feature_map = F.interpolate(
                feature_map,
                size=image_batch.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        cloud = self.gaussian_head(
            feature_map + self.images_merger(image_batch)
        )
        if has_view_axis:
            return GaussianCloud(
                xyz=cloud.xyz.reshape(batch, views, *cloud.xyz.shape[1:]),
                scales=cloud.scales.reshape(batch, views, *cloud.scales.shape[1:]),
                rotations=cloud.rotations.reshape(batch, views, *cloud.rotations.shape[1:]),
                opacities=cloud.opacities.reshape(batch, views, *cloud.opacities.shape[1:]),
                sh_coeffs=cloud.sh_coeffs.reshape(batch, views, *cloud.sh_coeffs.shape[1:]),
            )
        return cloud


__all__ = ["GSDPT"]

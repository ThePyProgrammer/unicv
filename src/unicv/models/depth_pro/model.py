"""DepthPro depth estimation model.

Implements the DepthPro architecture from:
  Bochkovskii et al. - Depth Pro: Sharp Monocular Metric Depth in Less Than a Second (2024)
  https://github.com/apple/ml-depth-pro

The model is registered as a ``VisionModule`` with:
  inputs:  RGB (single image)
  outputs: DEPTH
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicv.models.base import VisionModule
from unicv.models.depth_pro.encoder import DepthProEncoder
from unicv.nn.decoder import MultiresConvDecoder
from unicv.nn.fov import FOVNetwork
from unicv.utils.types import InputForm, Modality


class DepthPro(nn.Module):
    """DepthPro network (raw PyTorch module).

    Multi-scale patch-pyramid encoder → multi-resolution conv decoder →
    depth head (+ optional FoV head).
    """

    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        use_fov_head: bool = True,
        fov_encoder: nn.Module | None = None,
    ):
        """Initialise DepthPro.

        Args:
            encoder: The DepthProEncoder backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: ``(intermediate_dim, out_dim)`` for the final conv head.
            use_fov_head: Whether to attach the field-of-view estimation head.
            fov_encoder: Optional separate ViT encoder for FOV estimation.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        dim_decoder = decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        # Initialise final bias to zero.
        self.head[4].bias.data.fill_(0)  # type: ignore[union-attr]

        if use_fov_head:
            self.fov = FOVNetwork(num_features=dim_decoder, fov_encoder=fov_encoder)

    @property
    def img_size(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.img_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the encoder–decoder and return inverse depth + optional FOV.

        Args:
            x: Input image tensor, shape ``(B, 3, H, W)`` where
               ``H == W == self.img_size``.

        Returns:
            A tuple ``(canonical_inverse_depth, fov_deg)`` where
            *canonical_inverse_depth* has shape ``(B, 1, H, W)`` and
            *fov_deg* is ``None`` when the FOV head is disabled.
        """
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        encodings = self.encoder(x)
        features, features_0 = self.decoder(encodings)
        canonical_inverse_depth = self.head(features)

        fov_deg: torch.Tensor | None = None
        if hasattr(self, "fov"):
            fov_deg = self.fov(x, features_0.detach())

        return canonical_inverse_depth, fov_deg

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        f_px: torch.Tensor | None = None,
        interpolation_mode: str = "bilinear",
    ) -> dict[str, torch.Tensor]:
        """Infer metric depth and focal length for an input image.

        If the image is not at network resolution it is resized to
        ``(img_size × img_size)`` and the estimated depth is resized back.

        Args:
            x: Input image tensor, shape ``(B, 3, H, W)`` or ``(3, H, W)``.
            f_px: Optional known focal length in pixels. When provided the
                model's FOV estimate is ignored.
            interpolation_mode: Interpolation mode used for resize operations.

        Returns:
            Dict with keys ``"depth"`` (metric depth in metres) and
            ``"focallength_px"``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape
        resize = H != self.img_size or W != self.img_size

        if resize:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode=interpolation_mode,
                align_corners=False,
            )

        canonical_inverse_depth, fov_deg = self.forward(x)

        if f_px is None:
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))

        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = F.interpolate(
                inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
        return {"depth": depth.squeeze(), "focallength_px": f_px}


# ---------------------------------------------------------------------------
# VisionModule wrapper
# ---------------------------------------------------------------------------

class DepthProModel(VisionModule):
    """UniCV ``VisionModule`` wrapper around ``DepthPro``.

    Inputs:
        rgb (Modality.RGB, InputForm.SINGLE): RGB image tensor ``(B, 3, H, W)``.

    Outputs:
        Modality.DEPTH: Metric depth map in metres.
    """

    input_spec: dict[Modality, InputForm] = {Modality.RGB: InputForm.SINGLE}
    output_modalities: list[Modality] = [Modality.DEPTH]

    def __init__(self, net: DepthPro):
        """Initialise DepthProModel.

        Args:
            net: A pre-constructed ``DepthPro`` nn.Module instance.
        """
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        """Run inference and return a depth map.

        Args:
            **inputs: Must contain key ``"rgb"`` with tensor value.

        Returns:
            ``{Modality.DEPTH: depth_tensor}``
        """
        rgb: torch.Tensor = inputs[Modality.RGB.value]
        result = self.net.infer(rgb)
        return {Modality.DEPTH: result["depth"]}


__all__ = ["DepthPro", "DepthProModel"]
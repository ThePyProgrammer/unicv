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

    @classmethod
    def from_pretrained(
        cls,
        use_fov_head: bool = True,
        cache_dir: str | None = None,
    ) -> "DepthProModel":
        """Load the official Apple DepthPro pretrained weights.

        Downloads ``depth_pro.pt`` from ``apple/DepthPro`` on Hugging Face Hub
        and loads it into a freshly constructed ``DepthProModel``.

        The checkpoint was produced by Apple's original training code whose
        architecture is identical to this unicv reimplementation, so all
        keys load directly with no remapping.  ``strict=False`` is used to
        gracefully handle minor structural differences (e.g. the optional FOV
        head).

        Requirements
        ------------
        ``pip install huggingface_hub timm``

        Args:
            use_fov_head: Whether to include the FOV estimation head.  The
                official checkpoint always contains FOV weights; set this to
                ``False`` only if you want to discard them.
            cache_dir: Optional directory for the Hugging Face cache.  Defaults
                to the standard HF cache location (``~/.cache/huggingface``).

        Returns:
            A ``DepthProModel`` with pretrained weights loaded.

        Example::

            model = DepthProModel.from_pretrained()
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "DepthProModel.from_pretrained requires huggingface_hub.\n"
                "Install it with:  pip install huggingface_hub"
            )
        try:
            import timm
        except ImportError:
            raise ImportError(
                "DepthProModel.from_pretrained requires timm.\n"
                "Install it with:  pip install timm"
            )

        # --- Download checkpoint ---
        ckpt_path = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            cache_dir=cache_dir,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # --- Build architecture matching the official training config ---
        # ViT-L/16 as patch encoder (embed_dim=1024, 24 blocks)
        patch_encoder = timm.create_model(
            "vit_large_patch16_384",
            pretrained=False,
            num_classes=0,
            img_size=384,
        )
        # ViT-B/16 as image encoder (embed_dim=768, 12 blocks)
        image_encoder = timm.create_model(
            "vit_base_patch16_384",
            pretrained=False,
            num_classes=0,
            img_size=384,
        )
        encoder = DepthProEncoder(
            dims_encoder=[256, 512, 1024, 1024],
            patch_encoder=patch_encoder,
            image_encoder=image_encoder,
            hook_block_ids=[5, 11],      # ~1/4 and ~1/2 depth of ViT-L (24 blocks)
            decoder_features=256,
        )
        decoder = MultiresConvDecoder(
            dims_encoder=[256, 256, 512, 1024, 1024],
            dim_decoder=256,
        )
        net = DepthPro(
            encoder=encoder,
            decoder=decoder,
            last_dims=(32, 1),
            use_fov_head=use_fov_head,
        )

        # --- Load weights ---
        # Checkpoint keys match the DepthPro module directly (same codebase).
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing:
            import warnings
            shown = missing[:5]
            warnings.warn(
                f"DepthPro: {len(missing)} missing key(s) when loading pretrained "
                f"weights (first 5): {shown}",
                stacklevel=2,
            )

        return cls(net=net)


__all__ = ["DepthPro", "DepthProModel"]
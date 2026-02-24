"""Simple Depth Transformer (SDT) head.

Implements the SDT architecture from AnyDepth:
  AIGeeksGroup/AnyDepth - AnyDepth: Depth Estimation Made Easy
  https://github.com/AIGeeksGroup/AnyDepth

The SDT is a compact, efficient decoder for monocular depth estimation that
operates on multi-scale ViT features. Unlike the full DPT decoder, it avoids
readout tokens and uses a lightweight cross-scale attention + upsampling
scheme, achieving comparable accuracy with significantly fewer parameters.

Architecture summary
--------------------
For each scale level the SDT:
1. Linearly projects the flat patch tokens to a fixed channel width.
2. Applies a single self-attention layer to model spatial context.
3. Reshapes tokens into a 2-D feature map.
4. Fuses adjacent levels via a lightweight convolutional merge.
5. Progressively upsamples to the target resolution.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TokenAttention(nn.Module):
    """Single multi-head self-attention block operating on flat patch tokens."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual self-attention.

        Args:
            x: Token sequence ``(B, N, D)``.

        Returns:
            Updated token sequence ``(B, N, D)``.
        """
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        return x + attn_out


class _LevelProjector(nn.Module):
    """Projects one encoder level into the shared decoder width and attends."""

    def __init__(self, embed_dim: int, decoder_dim: int, num_heads: int, img_size: int, patch_size: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.attn = _TokenAttention(decoder_dim, num_heads)
        self.norm = nn.LayerNorm(decoder_dim)
        self.h = self.w = img_size // patch_size

    def forward(self, tokens: torch.Tensor, num_register_tokens: int = 0) -> torch.Tensor:
        """Project and reshape tokens to ``(B, D, h, w)``."""
        # Drop CLS + register tokens.
        tokens = tokens[:, 1 + num_register_tokens:, :]
        tokens = self.proj(tokens)
        tokens = self.attn(tokens)
        tokens = self.norm(tokens)
        B, _, D = tokens.shape
        return tokens.permute(0, 2, 1).reshape(B, D, self.h, self.w)


class _ConvFuse(nn.Module):
    """1×1 conv followed by 3×3 depth-wise conv for level fusion."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SDTHead(nn.Module):
    """Simple Depth Transformer decoder head (AnyDepth).

    Takes ``num_levels`` intermediate ViT hidden states and decodes them into
    a dense inverse-depth map at the input image resolution.

    Compared to the DPT decoder this head:
    - Uses a single attention layer per level (not full transformer blocks).
    - Fuses levels with lightweight 1×1 + depth-wise 3×3 convolutions.
    - Has ~60 % fewer parameters for the same ``decoder_dim``.
    """

    def __init__(
        self,
        embed_dim: int,
        decoder_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 8,
        patch_size: int = 14,
        img_size: int = 518,
        num_register_tokens: int = 0,
        out_channels: int = 1,
    ):
        """Initialise SDTHead.

        Args:
            embed_dim: Hidden dimension of the ViT backbone.
            decoder_dim: Uniform channel width inside the decoder.
            num_levels: Number of encoder levels to consume (typically 4).
            num_heads: Number of attention heads in the token attention blocks.
            patch_size: ViT patch size.
            img_size: Input image resolution (assumed square).
            num_register_tokens: Extra register tokens (4 for DINOv2).
            out_channels: Output channels for the prediction head (1 for depth).
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_register_tokens = num_register_tokens

        # Per-level projection + attention.
        self.level_projectors = nn.ModuleList([
            _LevelProjector(embed_dim, decoder_dim, num_heads, img_size, patch_size)
            for _ in range(num_levels)
        ])

        # Cross-level fusion: fuse adjacent up-sampled levels.
        # We iterate from coarsest (last) to finest (0).
        self.fuse_blocks = nn.ModuleList([
            _ConvFuse(decoder_dim * 2 if i < num_levels - 1 else decoder_dim, decoder_dim)
            for i in range(num_levels)
        ])

        # Final prediction head (×4 upsample → input resolution / patch_size → full res).
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=patch_size // 2, mode="bilinear", align_corners=True),
            nn.Conv2d(decoder_dim // 2, decoder_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(decoder_dim // 4, out_channels, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale ViT hidden states into a dense depth map.

        Args:
            hidden_states: List of ``num_levels`` tensors each of shape
                ``(B, 1 + num_register_tokens + N, D)``, ordered
                shallower → deeper.

        Returns:
            Predicted depth / inverse-depth map at the input resolution,
            shape ``(B, out_channels, H, W)``.
        """
        assert len(hidden_states) == self.num_levels, (
            f"Expected {self.num_levels} hidden states, got {len(hidden_states)}"
        )

        # Step 1 – project each level to a spatial feature map.
        spatial: list[torch.Tensor] = [
            proj(h, self.num_register_tokens)
            for proj, h in zip(self.level_projectors, hidden_states)
        ]

        # Step 2 – fuse from coarsest to finest.
        # Coarsest level (last) is upsampled to the next level's size, then
        # concatenated and fused.
        x = self.fuse_blocks[-1](spatial[-1])
        for i in range(self.num_levels - 2, -1, -1):
            target_h, target_w = spatial[i].shape[-2], spatial[i].shape[-1]
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
            x = torch.cat([x, spatial[i]], dim=1)
            x = self.fuse_blocks[i](x)

        return self.head(x)


__all__ = ["SDTHead"]

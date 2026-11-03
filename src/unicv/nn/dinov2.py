"""DINOv2 backbone wrapper.

Thin wrapper around ``torch.hub`` DINOv2 models that captures intermediate
hidden states for use by downstream decoders (DPT, SDT, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    """Thin wrapper around a ``torch.hub`` DINOv2 model.

    Returns the intermediate hidden states from a configurable set of
    transformer layers so that the DPT decoder can reassemble them.
    """

    # Map config name -> hub model name.
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
        hook_layer_ids: list[int] | None = None,
        num_register_tokens: int = 0,
        alt_start: int = -1,
        cat_token: bool = True,
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
            alt_start: First layer using alternating local/cross-view attention.
                ``-1`` disables cross-view attention.
            cat_token: Concatenate local and cross-view block features when
                returning intermediate states.
        """
        super().__init__()

        if variant not in self._HUB_NAMES:
            raise ValueError(
                f"variant must be one of {list(self._HUB_NAMES)}, got {variant!r}"
            )
        hub_name = self._HUB_NAMES[variant]
        self.embed_dim: int = self._EMBED_DIMS[variant]
        self.num_register_tokens = num_register_tokens

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=pretrained,
        )

        total_layers = len(self.model.blocks)
        if alt_start != -1 and not 0 <= alt_start < total_layers:
            raise ValueError(
                f"alt_start must be -1 or in [0, {total_layers}), got {alt_start}"
            )
        self.alt_start = alt_start
        self.cat_token = cat_token
        if self.alt_start != -1:
            self.model.camera_token = nn.Parameter(torch.randn(1, 2, self.embed_dim))

        if hook_layer_ids is None:
            # Default: sample 4 layers evenly starting from 1/4 depth.
            hook_layer_ids = [
                total_layers // 4 - 1,
                total_layers // 2 - 1,
                3 * total_layers // 4 - 1,
                total_layers - 1,
            ]

        self.hook_layer_ids: list[int] = hook_layer_ids

    def enable_cross_view_attention(self, alt_start: int, cat_token: bool = True) -> None:
        """Enable camera-token local/global attention at ``alt_start``."""
        total_layers = len(self.model.blocks)
        if not 0 <= alt_start < total_layers:
            raise ValueError(
                f"alt_start must be in [0, {total_layers}), got {alt_start}"
            )
        self.alt_start = alt_start
        self.cat_token = cat_token
        if not hasattr(self.model, "camera_token"):
            self.model.camera_token = nn.Parameter(torch.randn(1, 2, self.embed_dim))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return hidden states with local and optional cross-view attention.

        Args:
            x: Images shaped ``(B, 3, H, W)`` or a view set
                ``(B, S, 3, H, W)``.

        Returns:
            List of hidden-state tensors in ``hook_layer_ids`` order, each
            shaped ``(B, 1 + num_register_tokens + N, D)`` for single images,
            or ``(B, S, 1 + num_register_tokens + N, D)`` for view sets.
            When cross-view attention and ``cat_token`` are enabled, the final
            feature dimension is ``2 * D``.
        """
        if x.ndim == 4:
            images = x.unsqueeze(1)
            single_view = True
        elif x.ndim == 5:
            images = x
            single_view = False
        else:
            raise ValueError(
                f"Expected images shaped (B, 3, H, W) or (B, S, 3, H, W), got {tuple(x.shape)}"
            )

        batch, views, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB images, got {channels} channels")

        flat_images = images.reshape(batch * views, channels, height, width)
        tokens = self.model.prepare_tokens_with_masks(flat_images)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise TypeError(
                "DINOv2 prepare_tokens_with_masks must return a (B, N, D) tensor"
            )
        tokens = tokens.reshape(batch, views, *tokens.shape[1:])

        outputs: dict[int, torch.Tensor] = {}
        local_tokens = tokens
        for layer_id, block in enumerate(self.model.blocks):
            if self.alt_start != -1 and layer_id == self.alt_start:
                camera_tokens = self.model.camera_token
                reference = camera_tokens[:, :1].expand(batch, 1, -1)
                sources = camera_tokens[:, 1:].expand(batch, views - 1, -1)
                tokens = tokens.clone()
                tokens[:, :, 0] = torch.cat((reference, sources), dim=1)

            if self.alt_start != -1 and layer_id >= self.alt_start and layer_id % 2 == 1:
                block_input = tokens.reshape(batch, views * tokens.shape[2], tokens.shape[3])
                block_output = block(block_input)
                tokens = block_output.reshape(
                    batch, views, tokens.shape[2], tokens.shape[3]
                )
            else:
                block_input = tokens.reshape(batch * views, *tokens.shape[2:])
                block_output = block(block_input)
                tokens = block_output.reshape(batch, views, *block_output.shape[1:])
                local_tokens = tokens

            if layer_id in self.hook_layer_ids:
                hidden = (
                    torch.cat((local_tokens, tokens), dim=-1)
                    if self.cat_token and self.alt_start != -1
                    else tokens
                )
                outputs[layer_id] = hidden[:, 0] if single_view else hidden

        return [outputs[layer_id] for layer_id in self.hook_layer_ids]


__all__ = ["DINOv2Backbone"]

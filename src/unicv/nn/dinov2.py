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

    # Map config name → hub model name.
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
        """
        super().__init__()

        hub_name = self._HUB_NAMES[variant]
        self.embed_dim: int = self._EMBED_DIMS[variant]
        self.num_register_tokens = num_register_tokens

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=pretrained,
        )

        total_layers = len(self.model.blocks)
        if hook_layer_ids is None:
            # Default: sample 4 layers evenly starting from 1/4 depth.
            hook_layer_ids = [
                total_layers // 4 - 1,
                total_layers // 2 - 1,
                3 * total_layers // 4 - 1,
                total_layers - 1,
            ]

        self.hook_layer_ids: list[int] = hook_layer_ids
        self._features: dict[int, torch.Tensor] = {}

        # Register forward hooks on the selected transformer blocks.
        for layer_id in self.hook_layer_ids:
            self.model.blocks[layer_id].register_forward_hook(
                self._make_hook(layer_id)
            )

    def _make_hook(self, layer_id: int):
        def _hook(module, input, output):
            self._features[layer_id] = output
        return _hook

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward through DINOv2 and collect intermediate hidden states.

        Args:
            x: Input images, shape ``(B, 3, H, W)``.

        Returns:
            List of hidden-state tensors in ``hook_layer_ids`` order, each
            shape ``(B, 1 + num_register_tokens + N, D)``.
        """
        self._features.clear()
        _ = self.model(x)
        return [self._features[lid] for lid in self.hook_layer_ids]


__all__ = ["DINOv2Backbone"]

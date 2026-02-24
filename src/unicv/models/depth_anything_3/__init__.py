"""Depth Anything 3 model package.

Contains the Depth Anything 3 depth estimation model:
- ``DINOv2Backbone`` – DINOv2 ViT backbone with intermediate hook capture
- ``DepthAnything3`` – the raw PyTorch nn.Module
- ``DepthAnything3Model`` – VisionModule wrapper for use in unicv pipelines
"""

from unicv.models.depth_anything_3.model import (
    DINOv2Backbone,
    DepthAnything3,
    DepthAnything3Model,
)

__all__ = [
    "DINOv2Backbone",
    "DepthAnything3",
    "DepthAnything3Model",
]

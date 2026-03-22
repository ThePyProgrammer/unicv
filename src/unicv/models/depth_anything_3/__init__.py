"""Depth Anything 3 model package.

Contains the Depth Anything 3 depth estimation model:
- ``DepthAnything3`` -- the raw PyTorch nn.Module
- ``DepthAnything3Model`` -- VisionModule wrapper for use in unicv pipelines

Note: ``DINOv2Backbone`` has moved to ``unicv.nn.dinov2``.
"""

from unicv.models.depth_anything_3.model import (
    DepthAnything3,
    DepthAnything3Model,
)

__all__ = [
    "DepthAnything3",
    "DepthAnything3Model",
]

"""DepthPro model package.

Contains the full DepthPro depth estimation model:
- ``DepthProEncoder`` – multi-resolution patch pyramid encoder
- ``DepthPro`` – the raw PyTorch nn.Module
- ``DepthProModel`` – VisionModule wrapper for use in unicv pipelines
"""

from unicv.models.depth_pro.encoder import DepthProEncoder
from unicv.models.depth_pro.model import DepthPro, DepthProModel

__all__ = [
    "DepthProEncoder",
    "DepthPro",
    "DepthProModel",
]

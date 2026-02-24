"""Camera Depth Model (CDM) package.

Contains the CDM dual-ViT depth estimation model:
- ``TokenFusion``         – per-level additive RGB × depth token fusion
- ``CDM``                 – the raw PyTorch nn.Module
- ``CameraDepthModel``    – VisionModule wrapper (RGB + DEPTH → DEPTH)
"""

from unicv.models.cdm.model import (
    TokenFusion,
    CDM,
    CameraDepthModel,
)

__all__ = [
    "TokenFusion",
    "CDM",
    "CameraDepthModel",
]

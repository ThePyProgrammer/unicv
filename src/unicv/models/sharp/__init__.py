"""SHARP single-image Gaussian splat model package.

Contains:
- ``SHARP``       – the raw PyTorch nn.Module
- ``SHARPModel``  – VisionModule wrapper (RGB → SPLAT)
"""

from unicv.models.sharp.model import SHARP, SHARPModel

__all__ = ["SHARP", "SHARPModel"]

"""unicv.nn – neural network building blocks for computer vision models."""

from unicv.nn.cost_volume import PlaneSweepCostVolume
from unicv.nn.decoder import FeatureFusionBlock2d, MultiresConvDecoder, ResidualBlock
from unicv.nn.dpt import DPTDecoder, FeatureFusionBlock, Reassemble, ResidualConvUnit
from unicv.nn.fov import FOVNetwork
from unicv.nn.gaussian import GaussianHead
from unicv.nn.geometry import backproject_depth, homography_warp
from unicv.nn.sdt import SDTHead
from unicv.nn.sparse3d import (
    SPARSE3D_BACKEND_AVAILABLE,
    SparseConv3d,
    SparseVoxelTensor,
    devoxelize,
    voxelize,
)

__all__ = [
    # DPT decoder components
    "DPTDecoder",
    "FeatureFusionBlock",
    "Reassemble",
    "ResidualConvUnit",
    # DepthPro-style multires decoder
    "MultiresConvDecoder",
    "FeatureFusionBlock2d",
    "ResidualBlock",
    # FOV head
    "FOVNetwork",
    # SDT head (AnyDepth)
    "SDTHead",
    # Gaussian splat head
    "GaussianHead",
    # Camera-geometry utilities
    "backproject_depth",
    "homography_warp",
    # Plane-sweep cost volume
    "PlaneSweepCostVolume",
    # Sparse 3-D convolution
    "SPARSE3D_BACKEND_AVAILABLE",
    "SparseConv3d",
    "SparseVoxelTensor",
    "voxelize",
    "devoxelize",
]

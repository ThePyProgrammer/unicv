# Building Blocks (`unicv.nn`)

The `unicv.nn` package contains reusable neural network modules shared across multiple models. Nothing in this package knows about `VisionModule` or any specific model — it is pure PyTorch.

## Backbones

### DINOv2Backbone (`dinov2.py`)

Wraps a [DINOv2](https://github.com/facebookresearch/dinov2) ViT loaded via `torch.hub`. Registers forward hooks on selected transformer blocks to capture intermediate hidden states.

Shared by: Depth Anything 3, CDM, SHARP.

**Variants:** `vit_s` (384), `vit_b` (768), `vit_l` (1024), `vit_g` (1536).

**Output:** List of tensors `(B, 1 + num_register_tokens + N, D)`, one per hooked layer. By default, hooks are placed at 1/4, 1/2, 3/4, and full depth of the transformer.

## Decoders

### DPTDecoder (`dpt.py`)

Implements the [Dense Prediction Transformer](https://arxiv.org/abs/2103.13413) decoder. The workhorse decoder for most DINOv2-based models.

**Algorithm:**

1. **Reassemble** — For each hooked encoder layer, `Reassemble` strips the CLS/register tokens, reshapes the flat token sequence into a 2D spatial grid, projects to a uniform channel width, and optionally resamples spatially (4x, 2x, 1x, or 0.5x).

2. **Fuse** — Starting from the coarsest level, `FeatureFusionBlock` progressively merges adjacent scales. Each block applies a residual convolution to the finer features, adds them to the coarser (upsampled) features, then runs a second residual conv and 2x bilinear upsampling.

3. **Head** — A final conv sequence projects to the target output channels (1 for depth, 256 for dense features).

**Key types:** `Reassemble`, `FeatureFusionBlock`, `ResidualConvUnit`.

### MultiresConvDecoder (`decoder.py`)

An alternative multi-resolution decoder used by DepthPro. Takes encoder features from finest to coarsest and fuses them via `FeatureFusionBlock2d` blocks with deconv upsampling.

Returns `(features, lowres_features)` — the main output and the low-resolution feature map used by the FOV head.

## Prediction heads

### GaussianHead (`gaussian.py`)

Maps a dense feature map `(B, F, H, W)` to a `GaussianCloud` with `N = H x W` Gaussians per image. Regresses per-pixel:

- **xyz** — 3D positions (placeholder; replaced by backprojection in SHARP)
- **scales** — log-space, exponentiated to ensure positivity
- **rotations** — quaternions, L2-normalised to unit length
- **opacities** — sigmoid-activated to [0, 1]
- **sh_coeffs** — spherical harmonic coefficients for view-dependent colour

### FOVNetwork (`fov.py`)

Estimates a scalar field-of-view from an image and low-resolution encoder features. A cascade of stride-2 convolutions reduces spatial dimensions to a single scalar. Used by DepthPro to convert canonical inverse depth to metric depth.

### SDTHead (`sdt.py`)

The Simple Depth Transformer head from [AnyDepth](https://github.com/AIGeeksGroup/AnyDepth). For each encoder level:

1. Linear projection to a fixed channel width
2. Single-head self-attention for spatial context
3. Reshape to 2D feature map

Levels are then fused from coarsest to finest via `_ConvFuse` blocks (1x1 conv + depthwise 3x3 + BatchNorm + GELU).

## Geometry utilities (`geometry.py`)

### `backproject_depth(depth, K)`

Lifts a depth map `(B, 1, H, W)` to a 3D point cloud `(B, H, W, 3)` in camera coordinates using the pinhole intrinsics matrix `K (B, 3, 3)`.

**Algorithm:** Constructs a pixel-coordinate grid, applies `K_inv` to get ray directions, then scales by depth.

### `homography_warp(src_feats, depth_hyp, K_src, K_ref, E)`

Warps a source feature map to the reference view at a fronto-parallel depth plane. Used by the plane-sweep cost volume.

**Algorithm:** For a given depth hypothesis `d`, computes the 3x3 homography `H = K_src @ (R - t*n^T/d) @ K_ref_inv`, applies it to a reference pixel grid, and samples the source features via `grid_sample`.

### `default_intrinsics(B, H, W, device)`

Constructs a default pinhole camera matrix with `focal = max(H, W)` and principal point at image centre. Used by SHARP and SimpleRecon when no calibration data is available.

## Cost volume (`cost_volume.py`)

### PlaneSweepCostVolume

Constructs a `(B, D, H, W)` matching-similarity volume from multi-view features. For each of `D` depth hypotheses, warps source features to the reference view via `homography_warp` and computes normalised cross-correlation (NCC) similarity.

Depth hypotheses can be linearly or logarithmically spaced between `min_depth` and `max_depth`.

## Sparse 3D convolution (`sparse3d.py`)

### SparseConv3d

Drop-in replacement for `nn.Conv3d` that operates on sparse voxel grids. Auto-detects the available backend at import time:

1. `spconv` (preferred)
2. `MinkowskiEngine` (fallback)
3. Dense `nn.Conv3d` (if neither is installed)

Also provides `voxelize` / `devoxelize` for converting between dense tensors and `SparseVoxelTensor`.

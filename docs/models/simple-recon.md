# SimpleRecon

> Sayed et al. — *SimpleRecon: 3D Reconstruction Without 3D Convolutions* (Niantic, 2022)
>
> [Project](https://nianticlabs.github.io/simplerecon/)

**Input:** RGB frames (temporal) `list[(B, 3, H, W)]` | **Output:** Depth map `(B, 1, H, W)`

## Overview

SimpleRecon estimates depth for a reference frame using evidence from multiple source frames. Instead of heavy 3D feature volumes, it uses a lightweight plane-sweep cost volume with normalised cross-correlation — achieving competitive accuracy with a fraction of the compute.

This is the only multi-view model in UniCV and the only one using `InputForm.TEMPORAL`.

## Architecture

```
Frames: [ref, src_1, src_2, ...]
  │
  ├─ SimpleEncoder (per-frame CNN, stride 8)
  │   ├─ ref_feats   (B, C, H/8, W/8)
  │   └─ src_feats[] (B, C, H/8, W/8) each
  │
  ├─ PlaneSweepCostVolume
  │   ├─ For each depth hypothesis d in [d_min, ..., d_max]:
  │   │   ├─ Warp each source feature map to reference view
  │   │   │   via homography at depth d
  │   │   └─ Compute NCC similarity with reference features
  │   └─ Output: matching volume (B, D, H/8, W/8)
  │
  ├─ CostVolumeRegularizer (3D CNN)
  │   └─ Smoothed volume (B, D, H/8, W/8)
  │
  ├─ Soft argmin
  │   ├─ softmax over depth dimension → probabilities
  │   └─ weighted sum with depth hypotheses → expected depth
  │
  └─ Bilinear upsample to (B, 1, H, W)
```

## Algorithm walkthrough

1. **Encode.** A lightweight CNN (`SimpleEncoder`) with three stride-2 convolutions reduces each frame to a feature map at 1/8 resolution. The encoder is deliberately shallow to keep cost-volume construction memory-efficient.

2. **Build cost volume.** The `PlaneSweepCostVolume` module defines `D` depth hypotheses (linearly or logarithmically spaced between `min_depth` and `max_depth`). For each hypothesis:
   - Compute the depth-induced homography from each source view to the reference view using camera intrinsics and relative poses
   - Warp each source feature map to the reference frame via `homography_warp`
   - Compute normalised cross-correlation (NCC) between warped source and reference features
   - Average across source views

   The result is a `(B, D, H/8, W/8)` volume where each "slice" measures how well the multi-view evidence supports that depth value.

3. **Regularise.** A small 3D CNN (`CostVolumeRegularizer`) smooths the volume across both spatial and depth dimensions. The volume is treated as 5D `(B, 1, D, H, W)` for standard `Conv3d` compatibility. This fills in textureless regions and reduces photometric noise.

4. **Soft argmin.** A softmax over the depth dimension converts the regularised volume to a probability distribution. The expected depth is computed as the weighted sum of depth hypotheses:

   ```
   depth = sum(softmax(volume) * depth_hypotheses)
   ```

5. **Upsample.** Bilinear interpolation restores the depth map to the original input resolution.

## Camera parameters

The raw `SimpleRecon` module requires explicit camera parameters:
- `intrinsics (B, 3, 3)` — reference camera K matrix
- `src_intrinsics [list of (B, 3, 3)]` — per-source K matrices
- `src_poses [list of (B, 4, 4)]` — relative poses T_{src←ref}

The `SimpleReconModel` VisionModule wrapper defaults to identity poses and estimated intrinsics (via `default_intrinsics`). For real calibrated data, use the `SimpleRecon` module directly.

## UniCV classes

| Class | Type | Role |
|-------|------|------|
| `SimpleEncoder` | `nn.Module` | Per-frame CNN feature extractor |
| `CostVolumeRegularizer` | `nn.Module` | 3D CNN for volume smoothing |
| `SimpleRecon` | `nn.Module` | Full plane-sweep stereo pipeline |
| `SimpleReconModel` | `VisionModule` | UniCV wrapper (default camera params) |

## Pretrained weights

No public pretrained checkpoint is available for this model. Use `SimpleRecon.from_config()` to build the architecture for training.

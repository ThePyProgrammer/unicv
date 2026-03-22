# SHARP

> Apple — *SHARP: Single-image 3D Gaussian Reconstruction* (2024)
>
> [Project](https://apple.github.io/ml-sharp/) &#183; [arXiv 2512.10685](https://arxiv.org/abs/2512.10685)

**Input:** RGB image `(B, 3, H, W)` | **Output:** `GaussianCloud` with `N = H x W` Gaussians

## Overview

SHARP is a fully feed-forward model that converts a single RGB image into a dense 3D Gaussian splat representation in under one second. Unlike optimisation-based methods (3DGS), SHARP predicts all Gaussian parameters in a single forward pass — no iterative refinement needed.

## Architecture

```
Input (B, 3, H, W)
  │
  ├─ DINOv2 Backbone (ViT-L/14)
  │   └─ 4 intermediate hidden states
  │
  ├─ DPTDecoder (out_channels = features)
  │   └─ Dense feature map (B, F, H, W)
  │
  ├── GaussianHead ──────────────────────┐
  │   ├─ xyz_head    → (B, N, 3)        │
  │   ├─ scale_head  → (B, N, 3)  exp() │
  │   ├─ rot_head    → (B, N, 4)  norm  │  → GaussianCloud
  │   ├─ opacity_head→ (B, N, 1)  σ()   │
  │   └─ sh_head     → (B, N, C)        │
  │                                      │
  ├── Depth Head ────────────────────┐   │
  │   └─ features → softplus depth   │   │
  │      (B, 1, H, W)               │   │
  │                                  │   │
  ├── Backprojection ────────────────┘   │
  │   depth + K → xyz (B, N, 3)    ─────┘
  │   (replaces GaussianHead's xyz)
  │
  └─ GaussianCloud
```

## Algorithm walkthrough

1. **Encode.** A DINOv2 ViT encodes the input image, producing 4 multi-scale hidden states via forward hooks (same pattern as DA3 and CDM).

2. **Decode to feature map.** Unlike DA3 which decodes to a single depth channel, SHARP's `DPTDecoder` outputs a dense feature map with `features` channels (typically 256). This high-dimensional representation carries enough information for both geometry and appearance.

3. **Predict Gaussian parameters.** The `GaussianHead` applies separate 1x1 conv heads to the feature map, producing per-pixel:
   - **Scales:** 3 values in log-space, exponentiated to ensure positivity
   - **Rotations:** 4-component quaternions, L2-normalised to unit length
   - **Opacities:** sigmoid-activated to [0, 1]
   - **SH coefficients:** `(degree+1)^2 * 3` values for view-dependent colour

4. **Predict depth.** A shallow CNN head (1x1 conv → ReLU → 1x1 conv → softplus) predicts per-pixel depth from the same feature map. The softplus activation ensures depth > 0.

5. **Backproject.** Using the predicted depth and camera intrinsics `K`, `backproject_depth` lifts each pixel to a 3D point: `xyz = depth * K_inv @ pixel_coords`. This replaces the GaussianHead's raw `xyz` output with geometrically-grounded positions.

6. **Assemble.** A new `GaussianCloud` is constructed combining the backprojected `xyz` with the head-predicted scales, rotations, opacities, and SH coefficients.

## Camera intrinsics

The `SHARP` nn.Module accepts an optional `intrinsics` tensor `(B, 3, 3)`. When omitted, it defaults to a pinhole matrix with `focal = max(H, W)` and principal point at image centre (via `default_intrinsics`).

The `SHARPModel` VisionModule always uses the default. For metric-scale output, call `SHARP.forward(rgb, intrinsics=K)` directly.

## UniCV classes

| Class | Type | Role |
|-------|------|------|
| `SHARP` | `nn.Module` | Backbone + DPT + GaussianHead + depth + backprojection |
| `SHARPModel` | `VisionModule` | UniCV wrapper (default intrinsics) |

## Pretrained weights

```python
from unicv.models.sharp import SHARPModel

model = SHARPModel.from_pretrained()
cloud = model(rgb=image_tensor)[Modality.SPLAT]
```

Downloads from Apple's CDN. The official checkpoint uses a DepthPro-based encoder (not DINOv2), so only fusion-block conv weights transfer. Shape-mismatched keys are silently dropped.

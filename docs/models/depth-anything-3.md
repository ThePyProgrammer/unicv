# Depth Anything 3

> ByteDance-Seed — *Depth Anything 3* (2025)
>
> [Project](https://depth-anything-3.github.io/) &#183; [Code](https://github.com/ByteDance-Seed/Depth-Anything-3)

**Input:** RGB image `(B, 3, H, W)` or views `(B, S, 3, H, W)`
**Output:** Depth map(s), with an optional per-view Gaussian cloud for `vit_g`

## Overview

Depth Anything 3 (DA3) uses a DINOv2 backbone with alternating local and cross-view attention, followed by a DPT decoder. The backbone replaces each view's CLS token with a learned camera token at `alt_start`. The `vit_g` configuration also includes an image-conditioned GSDPT Gaussian branch.

## Architecture

```
Input (B, S, 3, H, W)   (S=1 for a single image)
  │
  ├─ DINOv2 Backbone (ViT-L/14)
  │   ├─ Local attention within each view
  │   ├─ Learned camera-token injection at alt_start
  │   └─ Alternating local and cross-view attention
  │
  ├─ DPTDecoder
  │   ├─ Reassemble: tokens → spatial maps at 4 scales
  │   ├─ Fuse: coarse → fine via FeatureFusionBlocks
  │   └─ Head: conv → 2x upsample → conv → ReLU
  │
  ├─ Output depth map(s)
  └─ GSDPT (vit_g): merge dense features with RGB, predict GaussianCloud
```

## Algorithm walkthrough

1. **Encode.** Each view is patch-embedded independently. After `alt_start`, a learned camera token replaces the CLS token and odd-indexed blocks attend across all views; intervening blocks retain per-view attention. Intermediate states include concatenated local and cross-view features.

2. **Reassemble.** The DPT decoder's `Reassemble` blocks strip the CLS and register tokens, reshape each view's patch tokens to a 2D grid, project to 256 channels, and resample to different spatial scales (4x, 2x, 1x, 0.5x relative to the patch grid).

3. **Fuse.** Starting from the coarsest scale (0.5x), `FeatureFusionBlock`s progressively merge features upward. Each block adds the finer features through a residual conv, applies a second residual conv, upsamples 2x, and projects via a 1x1 conv.

4. **Predict.** The depth head predicts a dense depth map for every view. When enabled, GSDPT fuses the dense DPT feature map with an RGB feature branch and predicts per-pixel Gaussian parameters.

5. **Resize.** If the decoder output doesn't match the input spatial dimensions, a final bilinear interpolation aligns them.

## Variants

| Variant  | Backbone       | Embed dim | Hub repo                  | Gaussian head |
|----------|----------------|-----------|---------------------------|---------------|
| `vit_s`  | DINOv2 ViT-S/14 | 384     | `depth-anything/DA3-SMALL` | No |
| `vit_b`  | DINOv2 ViT-B/14 | 768     | `depth-anything/DA3-BASE`  | No |
| `vit_l`  | DINOv2 ViT-L/14 | 1024    | `depth-anything/DA3-LARGE` | No |
| `vit_g`  | DINOv2 ViT-G/14 | 1536    | `depth-anything/DA3-GIANT` | Yes |

`DepthAnything3.from_config(..., include_gaussian_head=True)` enables GSDPT for other variants; `False` disables it for `vit_g`. The raw `DepthAnything3` module accepts `(B, S, 3, H, W)` for multi-view input. `DepthAnything3Model` returns both `Modality.DEPTH` and `Modality.SPLAT` when its network has a Gaussian head.

## UniCV classes

| Class | Type | Role |
|-------|------|------|
| `DepthAnything3` | `nn.Module` | Backbone + DPT decoder |
| `DepthAnything3Model` | `VisionModule` | UniCV wrapper |
| `GSDPT` | `nn.Module` | Image-conditioned per-view Gaussian decoder |

## Pretrained weights

```python
from unicv.models.depth_anything_3 import DepthAnything3Model

model = DepthAnything3Model.from_pretrained(variant="vit_l")
```

Checkpoints are in safetensors format. Keys are remapped from the official naming convention (e.g. `pretrained.*` → `backbone.model.*`, `depth_head.scratch.refinenet*` → `decoder.fusion_blocks.*`). The UniCV GSDPT module is newly initialized by `from_config`; loading a depth checkpoint does not provide pretrained Gaussian parameters.

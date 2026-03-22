# Depth Anything 3

> ByteDance-Seed — *Depth Anything 3* (2025)
>
> [Project](https://depth-anything-3.github.io/) &#183; [Code](https://github.com/ByteDance-Seed/Depth-Anything-3)

**Input:** RGB image `(B, 3, H, W)` | **Output:** Depth map `(B, 1, H, W)`

## Overview

Depth Anything 3 (DA3) is a monocular depth estimator built on a DINOv2 backbone with a DPT decoder. It is the simplest architecture in UniCV and serves as the canonical example of the DINOv2 + DPT pattern that CDM and SHARP also follow.

## Architecture

```
Input (B, 3, H, W)
  │
  ├─ DINOv2 Backbone (ViT-L/14)
  │   └─ Forward hooks capture hidden states at 4 layers
  │      [layer 5, 11, 17, 23] for ViT-L (24 blocks)
  │
  ├─ DPTDecoder
  │   ├─ Reassemble: tokens → spatial maps at 4 scales
  │   ├─ Fuse: coarse → fine via FeatureFusionBlocks
  │   └─ Head: conv → 2x upsample → conv → ReLU
  │
  └─ Output depth map (B, 1, H, W)
      └─ Bilinear upsample to input resolution if needed
```

## Algorithm walkthrough

1. **Encode.** The input image is passed through a DINOv2 ViT. Forward hooks on four evenly-spaced transformer blocks capture intermediate hidden states. Each state is a sequence of patch tokens `(B, 1 + N, D)` where `N = (H/14)^2`.

2. **Reassemble.** The DPT decoder's `Reassemble` blocks strip the CLS token, reshape each token sequence to a 2D spatial grid, project to 256 channels, and resample to different spatial scales (4x, 2x, 1x, 0.5x relative to the patch grid).

3. **Fuse.** Starting from the coarsest scale (0.5x), `FeatureFusionBlock`s progressively merge features upward. Each block adds the finer features through a residual conv, applies a second residual conv, upsamples 2x, and projects via a 1x1 conv.

4. **Predict.** A lightweight head (3x3 conv → 2x bilinear upsample → 1x1 conv → ReLU) produces a single-channel non-negative depth map.

5. **Resize.** If the decoder output doesn't match the input spatial dimensions, a final bilinear interpolation aligns them.

## Variants

| Variant  | Backbone       | Embed dim | Hub repo                  |
|----------|----------------|-----------|---------------------------|
| `vit_s`  | DINOv2 ViT-S/14 | 384     | `depth-anything/DA3-SMALL` |
| `vit_b`  | DINOv2 ViT-B/14 | 768     | `depth-anything/DA3-BASE`  |
| `vit_l`  | DINOv2 ViT-L/14 | 1024    | `depth-anything/DA3-LARGE` |
| `vit_g`  | DINOv2 ViT-G/14 | 1536    | `depth-anything/DA3-GIANT` |

## UniCV classes

| Class | Type | Role |
|-------|------|------|
| `DepthAnything3` | `nn.Module` | Backbone + DPT decoder |
| `DepthAnything3Model` | `VisionModule` | UniCV wrapper |

## Pretrained weights

```python
from unicv.models.depth_anything_3 import DepthAnything3Model

model = DepthAnything3Model.from_pretrained(variant="vit_l")
```

Checkpoints are in safetensors format. Keys are remapped from the official naming convention (e.g. `pretrained.*` → `backbone.model.*`, `depth_head.scratch.refinenet*` → `decoder.fusion_blocks.*`).

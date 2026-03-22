# DepthPro

> Bochkovskii et al. — *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second* (Apple, 2024)
>
> [Paper](https://arxiv.org/abs/2410.02073) &#183; [Code](https://github.com/apple/ml-depth-pro)

**Input:** RGB image `(B, 3, 1536, 1536)` | **Output:** Metric depth in metres `(B, 1, H, W)`

## Overview

DepthPro predicts sharp, metric monocular depth from a single image. Its key innovation is a multi-scale patch-pyramid encoder that processes the image at three resolutions simultaneously through a shared ViT backbone, producing features that capture both fine boundary detail and global scene structure.

## Architecture

```
Input (B, 3, 1536, 1536)
  │
  ├─ Image Pyramid (3 levels: 1536, 768, 384)
  │
  ├─ Sliding-window patches per level
  │   ├─ Level 0: 5x5 = 25 patches @ 384x384 (overlap 0.25)
  │   ├─ Level 1: 3x3 = 9  patches @ 384x384 (overlap 0.5)
  │   └─ Level 2: 1x1 = 1  patch   @ 384x384
  │
  ├─ Patch Encoder (ViT-L/16, shared)
  │   ├─ All 35 patches encoded in one batched forward pass
  │   └─ Hooks capture intermediate features at blocks [5, 11]
  │
  ├─ Merge patches back to spatial maps (with overlap trimming)
  │
  ├─ Image Encoder (ViT-B/16) on low-res level
  │   └─ Fused with Level 2 features via concat + 1x1 conv
  │
  ├─ Upsample all levels to matching spatial resolutions
  │
  ├─ MultiresConvDecoder (fuse finest → coarsest)
  │   └─ Returns (features, lowres_features)
  │
  ├─ Depth Head → canonical inverse depth (B, 1, H, W)
  │
  └─ [Optional] FOV Head → field-of-view in degrees
      └─ Converts inverse depth to metric depth via:
         depth = 1 / (inverse_depth * W / f_px)
```

## Encoder walkthrough

The `DepthProEncoder` is the most complex component. Its forward pass:

1. **Create pyramid.** Bilinear-downsample the input to three levels: 1x (1536), 0.5x (768), 0.25x (384).

2. **Split into patches.** Each level is tiled into overlapping 384x384 patches using a sliding window. The overlap varies per level to balance coverage and computation.

3. **Batch encode.** All patches from all levels (25 + 9 + 1 = 35 per image) are concatenated along the batch dimension and passed through the patch encoder ViT in a single forward call. Forward hooks on blocks 5 and 11 capture intermediate representations.

4. **Merge.** The encoded patch features are split back by level, then spatially merged by concatenating along height/width with overlap regions trimmed.

5. **Global context.** The lowest-resolution level is also passed through a separate image encoder ViT. Its output is fused with the Level 2 patch features.

6. **Upsample.** Each level's features are projected and upsampled (via learned ConvTranspose2d) to the target spatial dimensions for the decoder.

## Inference (`infer`)

The `infer` method wraps `forward` with practical conveniences:

- Accepts images at any resolution (auto-resizes to 1536x1536)
- Handles 3D `(3, H, W)` input by adding a batch dim
- Converts canonical inverse depth to metric depth using the FOV estimate (or a provided `f_px`)
- Resizes output back to the original resolution
- Returns `{"depth": ..., "focallength_px": ...}`

## UniCV classes

| Class | Type | Role |
|-------|------|------|
| `DepthProEncoder` | `nn.Module` | Multi-scale patch-pyramid encoder |
| `DepthPro` | `nn.Module` | Full encoder-decoder network |
| `DepthProModel` | `VisionModule` | UniCV wrapper (calls `infer`) |

## Pretrained weights

```python
from unicv.models.depth_pro import DepthProModel

model = DepthProModel.from_pretrained()
```

Downloads from [`apple/DepthPro`](https://huggingface.co/apple/DepthPro). Checkpoint keys match the unicv architecture directly (same codebase as Apple's original).

Requires: `pip install unicv[pretrained]`

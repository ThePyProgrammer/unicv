# Camera Depth Model (CDM)

> ByteDance — *Manipulation as in Simulation: Camera Depth Model* (2025)
>
> [Project](https://manipulation-as-in-simulation.github.io/#cdm-results)

**Input:** RGB `(B, 3, H, W)` + raw depth `(B, 1, H, W)` | **Output:** Refined depth `(B, 1, H, W)`

## Overview

CDM refines noisy depth maps from commodity depth cameras (RealSense, Kinect) using a dual-ViT architecture. One ViT branch encodes the RGB image, another encodes the raw depth signal. At each encoder level, the two token streams are fused and the result is decoded into a clean depth map.

The key insight is that RGB and depth provide complementary cues — RGB gives sharp edges and semantic context, while raw depth gives absolute scale despite noise. The per-level fusion lets the model learn which modality to trust at each spatial frequency.

## Architecture

```
RGB (B, 3, H, W)           Raw Depth (B, 1, H, W)
  │                            │
  │                         depth_proj (1→3 channels)
  │                            │
  ├─ DINOv2 Backbone ─┐  ┌─ DINOv2 Backbone ─┐
  │  (RGB branch)      │  │  (Depth branch)    │
  │                    │  │                    │
  │  4 hidden states   │  │  4 hidden states   │
  └────────────────────┘  └────────────────────┘
           │                        │
           └──── TokenFusion x4 ────┘
                      │
                 fused = rgb + proj(depth)
                      │
                 DPTDecoder
                      │
                 Depth map (B, 1, H, W)
```

## Algorithm walkthrough

1. **Project depth.** The single-channel raw depth is expanded to 3 channels via a learned 1x1 convolution, making it compatible with the DINOv2 backbone's expected input shape.

2. **Dual encoding.** Two independent DINOv2 backbones (same architecture, independent weights) process the RGB and projected-depth inputs. Each produces 4 intermediate hidden-state tensors via forward hooks.

3. **Token fusion.** At each of the 4 encoder levels, a `TokenFusion` module combines the two streams: `fused = rgb_tokens + Linear(depth_tokens)`. The additive residual connection preserves RGB gradient flow while injecting depth information.

4. **Decode.** The 4 fused hidden states are passed to a standard `DPTDecoder` which reassembles them into spatial feature maps and progressively fuses them to produce a single-channel depth prediction.

5. **Resize.** If the decoder output is coarser than the input, bilinear interpolation restores the original resolution.

## Camera variants

Each depth sensor has a distinct noise profile. CDM checkpoints are trained per sensor:

| Camera   | Sensor               | Hub repo                                  |
|----------|----------------------|-------------------------------------------|
| `d405`   | Intel RealSense D405 | `depth-anything/camera-depth-model-d405`  |
| `d435`   | Intel RealSense D435 | `depth-anything/camera-depth-model-d435`  |
| `l515`   | Intel RealSense L515 | `depth-anything/camera-depth-model-l515`  |
| `kinect` | Azure Kinect         | `depth-anything/camera-depth-model-kinect` |

## UniCV classes

| Class | Type | Role |
|-------|------|------|
| `TokenFusion` | `nn.Module` | Per-level additive RGB+depth fusion |
| `CDM` | `nn.Module` | Dual backbone + fusion + DPT decoder |
| `CameraDepthModel` | `VisionModule` | UniCV wrapper |

## Pretrained weights

```python
from unicv.models.cdm import CameraDepthModel

model = CameraDepthModel.from_pretrained(camera="d405")
result = model(rgb=rgb_tensor, depth=raw_depth_tensor)
```

Checkpoint keys are remapped from the official naming convention. The shared `_remap_dpt_key` helper handles the DPT decoder portion; CDM-specific prefixes (`pretrained.*`, `rgb_encoder.*`, `depth_encoder.*`) are handled locally.

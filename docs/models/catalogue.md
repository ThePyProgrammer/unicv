# Model Catalogue

## Implemented

| Model | Paper | Class | Input → Output | Pretrained |
|-------|-------|-------|----------------|------------|
| [DepthPro](https://github.com/apple/ml-depth-pro) | Apple, 2024 | `DepthProModel` | RGB → Depth | `DepthProModel.from_pretrained()` |
| [Depth Anything 3](https://depth-anything-3.github.io/) | ByteDance, 2025 | `DepthAnything3Model` | RGB → Depth | `DepthAnything3Model.from_pretrained(variant=...)` |
| [Camera Depth Model](https://manipulation-as-in-simulation.github.io/#cdm-results) | ByteDance, 2025 | `CameraDepthModel` | RGB + Depth → Depth | `CameraDepthModel.from_pretrained(camera=...)` |
| [SHARP](https://apple.github.io/ml-sharp/) | Apple, 2024 | `SHARPModel` | RGB → Splat | `SHARPModel.from_pretrained()` |
| [SimpleRecon](https://nianticlabs.github.io/simplerecon/) | Niantic, 2022 | `SimpleReconModel` | RGB (temporal) → Depth | -- |

## Planned

### Gaussian splat models

| Model | Status |
|-------|--------|
| [DepthSplat](https://haofeixu.github.io/depthsplat/) | Planned -- joint depth + splat from stereo/multi-view pairs |
| [InstantSplat](https://instantsplat.github.io/) | Planned -- pose-free pipeline via DUSt3R initialisation |
| [LongSplat](https://github.com/NVlabs/LongSplat) | Planned -- online real-time splats from long video |

### Mesh models

| Model | Status |
|-------|--------|
| [SuGaR](https://anttwo.github.io/sugar/) | Planned -- Gaussian-based surface reconstruction |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Planned -- diffusion-based single-image 3D generation |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Planned -- sparse 3D VAE + flow-matching transformer |

### Point cloud models

| Model | Status |
|-------|--------|
| [POMATO](https://github.com/wyddmw/POMATO) | Planned -- pose-aware multi-frame RGB → point cloud |
| [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/) | Planned -- monocular SLAM via DUSt3R matching transformer |

## Full modality matrix

| Model | Input | Form | Output |
|-------|-------|------|--------|
| DepthPro | RGB | Single | Depth |
| Depth Anything 3 | RGB | Single | Depth |
| Camera Depth Model | RGB + Depth | Single | Depth |
| SimpleRecon | RGB | Temporal | Depth |
| SHARP | RGB | Single | Splat |
| DepthSplat | RGB | List | Splat |
| InstantSplat | RGB | Temporal | Splat |
| LongSplat | RGB | Temporal | Splat |
| SuGaR | RGB | List | Mesh |
| Hunyuan3D-2.1 | RGB | Single | Mesh |
| TRELLIS.2 | RGB | Single | Mesh |
| POMATO | RGB | Temporal | Point Cloud |
| MASt3R-SLAM | RGB | Temporal | Point Cloud |

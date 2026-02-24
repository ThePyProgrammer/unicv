# UniCV

[![PyPI Latest Release](https://img.shields.io/pypi/v/unicv.svg?logo=python&logoColor=white&color=blue)](https://pypi.org/project/unicv/)
[![CI](https://img.shields.io/github/actions/workflow/status/aether-raid/unicv/ci.yml?label=CI&logo=github)](https://github.com/aether-raid/unicv/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

UniCV is a unified, extensible framework for computer vision models that operate across heterogeneous input and output representations. It wraps state-of-the-art models — depth estimators, Gaussian splat predictors, mesh generators, and more — behind a single, composable `VisionModule` interface.

---

## Philosophy

Modern computer vision has fragmented into dozens of incompatible APIs: each model ships with its own preprocessing, its own output format, and its own integration burden. 

The architecture and design philosophy of UniCV is inspired by modular deep learning ecosystems such as [`pytorch`](https://github.com/pytorch/pytorch) and HuggingFace's [`transformers`](https://github.com/huggingface/transformers), as well as recent efforts toward foundation models and generalist perception systems in computer vision. Rather than prescribing fixed pipelines (e.g. RGB → Depth or RGB → Mesh), UniCV abstracts vision algorithms as composable **transformations between representation spaces**.

The core abstraction of UniCV is `VisionModule`, which defines a standardized interface for mapping *any combination of visual input modalities* to *any combination of output modalities*. These modalities include, but are not limited to:

- RGB images
- Depth maps
- Point clouds
- Meshes
- Gaussian splats and other implicit or semi-implicit scene representations

Concrete vision algorithms—such as monocular depth estimation, RGB-to-point-cloud reconstruction, or RGB-D refinement—are implemented as subclasses of this abstract interface. Existing models available online (e.g. DepthPro, MiDaS, CDM, or point-cloud reconstruction networks) can be redefined within this framework without altering their internal logic, allowing them to be seamlessly integrated into a shared system.

UniCV is hence designed to accommodate the full spectrum of modern 3D perception: classical CNNs, ViT-based backbones, implicit neural representations, Gaussian splatting, and diffusion-based generation.

This abstraction enables UniCV to decouple **input modality**, **latent processing**, and **output representation**, encouraging reuse, composition, and extension of vision algorithms. Models may share encoders, latent spaces, or decoders, and can be combined or chained to support progressive or multi-stage reconstruction pipelines.

The conceptual motivation for UniCV is closely aligned with the emergence of **foundation models for perception**, where a single system is expected to reason across tasks, representations, and data sources. By enforcing a common interface at the representation level, UniCV facilitates cross-representation supervision, multi-task learning, and interoperability between otherwise incompatible vision methods.

## Motivation

UniCV aims to support vision systems that are:

1. **Modality-agnostic**, capable of ingesting arbitrary combinations of visual inputs (e.g. RGB, depth, point clouds) without architectural redesign.
2. **Representation-agnostic**, able to emit multiple scene representations from a shared latent abstraction.
3. **Algorithm-agnostic**, allowing existing and future computer vision models to be wrapped, extended, or replaced under a common interface.
4. **Composable**, enabling complex pipelines to be constructed by chaining or jointly training multiple `VisionModule`-based modules.
5. **Extensible**, supporting both classical CV algorithms and modern deep learning approaches, including implicit scene representations and neural rendering techniques.
6. **Foundation-ready**, serving as an architectural substrate for training large, generalist vision models capable of cross-task and cross-representation transfer.

In addition to standard convolutional and Transformer-based architectures, UniCV is designed to accommodate emerging paradigms such as implicit neural representations, Gaussian splatting, and hybrid geometric–neural pipelines, enabling a unified experimental platform for next-generation 3D perception systems.


---

## Architecture Overview

```
src/unicv/
├── utils/
│   └── types.py              # Modality and InputForm enums
├── models/
│   ├── base.py               # VisionModule abstract base class
│   ├── depth_pro/            # DepthPro (Apple, 2024)
│   ├── depth_anything_3/     # Depth Anything 3 (ByteDance, 2025)
│   ├── cdm/                  # Camera Depth Model (ByteDance, 2025)
│   ├── sharp/                # SHARP (Apple, 2024)
│   └── simplerecon/          # SimpleRecon (Niantic, 2022)
└── nn/
    ├── decoder.py            # MultiresConvDecoder, FeatureFusionBlock2d
    ├── dpt.py                # DPTDecoder, Reassemble, FeatureFusionBlock
    ├── fov.py                # FOVNetwork (field-of-view estimation)
    ├── sdt.py                # SDTHead (AnyDepth lightweight decoder)
    ├── gaussian.py           # GaussianHead
    └── geometry.py           # backproject_depth, homography_warp
```

### The `VisionModule` interface


Every model in UniCV inherits from the   `VisionModule` interface, which is defined by three key attributes and one method:

- `input_spec: dict[Modality, InputForm]` — declares what the model needs (e.g. a single RGB image, a temporal sequence of RGB frames).
- `output_modalities: list[Modality]` — declares what the model produces (e.g. a depth map, a Gaussian cloud).
- `forward(**inputs) -> dict[Modality, Any]` — the actual computation.

Calling an instance validates inputs, dispatches to `forward`, and validates outputs automatically. Models can be **chained**, **swapped**, or **jointly trained** without rewriting pipelines.

An example is shown below:

```python
from unicv.models.base import VisionModule
from unicv.utils.types import Modality, InputForm

class MyModel(VisionModule):
    input_spec         = {Modality.RGB: InputForm.SINGLE}
    output_modalities  = [Modality.DEPTH]

    def forward(self, **inputs):
        rgb   = inputs["rgb"]   # validated automatically
        depth = ...             # your model logic
        return {Modality.DEPTH: depth}

model  = MyModel()
result = model(rgb=image_tensor)   # → {Modality.DEPTH: tensor}
```

### `unicv.nn` building blocks

| Class                  | File          | Purpose                                                                                 |
| ---------------------- | ------------- | --------------------------------------------------------------------------------------- |
| `MultiresConvDecoder`  | `decoder.py`  | Fuses multi-scale encoder maps finest → coarsest; used by DepthPro                      |
| `FeatureFusionBlock2d` | `decoder.py`  | Single fusion step with optional residual skip and deconv upsampling                    |
| `DPTDecoder`           | `dpt.py`      | Full DPT pipeline: reassemble patch tokens → spatial maps, fuse; used by DA3 and CDM    |
| `Reassemble`           | `dpt.py`      | Converts flat ViT patch tokens to 2-D feature maps at a configurable scale              |
| `FeatureFusionBlock`   | `dpt.py`      | DPT-style fusion with 2× bilinear upsampling                                            |
| `FOVNetwork`           | `fov.py`      | Estimates a scalar field-of-view from a low-resolution feature map                      |
| `SDTHead`              | `sdt.py`      | Lightweight decoder from AnyDepth: per-level attention + depth-wise fusion              |
| `GaussianHead`         | `gaussian.py` | Regresses per-pixel Gaussian parameters (scales, rotations, opacities, SH coefficients) |

---

## Implemented Models

| Model                                                                              | Paper           | Class                 | Input → Output         | Pretrained                                         |
| ---------------------------------------------------------------------------------- | --------------- | --------------------- | ---------------------- | -------------------------------------------------- |
| [DepthPro](https://github.com/apple/ml-depth-pro)                                  | Apple, 2024     | `DepthProModel`       | RGB → Depth            | `DepthProModel.from_pretrained()`                  |
| [Depth Anything 3](https://depth-anything-3.github.io/)                            | ByteDance, 2025 | `DepthAnything3Model` | RGB → Depth            | `DepthAnything3Model.from_pretrained(variant=...)` |
| [Camera Depth Model](https://manipulation-as-in-simulation.github.io/#cdm-results) | ByteDance, 2025 | `CameraDepthModel`    | RGB + Depth → Depth    | `CameraDepthModel.from_pretrained(camera=...)`     |
| [SHARP](https://apple.github.io/ml-sharp/)                                         | Apple, 2024     | `SHARPModel`          | RGB → Splat            | `SHARPModel.from_pretrained()`                     |
| [SimpleRecon](https://nianticlabs.github.io/simplerecon/)                          | Niantic, 2022   | `SimpleReconModel`    | RGB (temporal) → Depth | —                                                  |

---

## Installation

### From PyPI

```bash
pip install unicv
```

### From source

```bash
git clone https://github.com/aether-raid/unicv.git
cd unicv
pip install .
```

### Development environment

Requires [uv](https://github.com/astral-sh/uv):

```bash
uv sync --dev
uv run pytest -v   # run the full test suite
```

---

## Loading Pretrained Weights

Each model with a `from_pretrained` classmethod downloads the official checkpoint and loads it into the UniCV architecture. Install the optional dependencies first:

```bash
pip install huggingface_hub timm safetensors
```

### DepthPro

Downloads Apple's DepthPro weights from [`apple/DepthPro`](https://huggingface.co/apple/DepthPro) on Hugging Face. Requires `huggingface_hub` and `timm`.

```python
from unicv.models.depth_pro import DepthProModel

model = DepthProModel.from_pretrained()               # includes FoV head
model = DepthProModel.from_pretrained(use_fov_head=False)   # depth-only

model.eval()
result = model(rgb=image_tensor)   # (B, 3, 1536, 1536)
depth  = result["depth"]           # metric depth in metres
```

### Depth Anything 3

Downloads from the [`depth-anything`](https://huggingface.co/depth-anything) organisation. Requires `huggingface_hub` and `safetensors`.

| `variant` | Hugging Face repo                      | Backbone embed dim |
| --------- | -------------------------------------- | ------------------ |
| `"vit_s"` | `depth-anything/DA3-SMALL`             | 384                |
| `"vit_b"` | `depth-anything/DA3-BASE`              | 768                |
| `"vit_l"` | `depth-anything/DA3-LARGE` *(default)* | 1024               |
| `"vit_g"` | `depth-anything/DA3-GIANT`             | 1536               |

```python
from unicv.models.depth_anything_3 import DepthAnything3Model

model = DepthAnything3Model.from_pretrained(variant="vit_l")
model.eval()

result = model(rgb=image_tensor)   # (B, 3, H, W)
depth  = result["depth"]           # inverse-depth, (B, 1, H, W)
```

### Camera Depth Model (CDM)

Downloads camera-specific checkpoints. Choose the variant that matches your depth sensor. Requires `huggingface_hub`.

| `camera`             | Sensor               |
| -------------------- | -------------------- |
| `"d405"` *(default)* | Intel RealSense D405 |
| `"d435"`             | Intel RealSense D435 |
| `"l515"`             | Intel RealSense L515 |
| `"kinect"`           | Azure Kinect         |

```python
from unicv.models.cdm import CameraDepthModel

model  = CameraDepthModel.from_pretrained(camera="d405")
model.eval()

result  = model(rgb=rgb_tensor, depth=raw_depth_tensor)
refined = result["depth"]   # (B, 1, H, W)
```

### SHARP

Downloads directly from Apple's CDN — no Hugging Face dependency required.

```python
from unicv.models.sharp import SHARPModel

model = SHARPModel.from_pretrained()
model.eval()

result = model(rgb=image_tensor)   # (B, 3, H, W)
cloud  = result["splat"]           # GaussianCloud with N = H×W Gaussians
```

> **Note** — the official SHARP checkpoint uses a DepthPro-based encoder
> (`SlidingPyramidNetwork` + `TimmViT`), not DINOv2. UniCV applies best-effort
> partial remappings for fusion-block conv weights; backbone weights are not
> transferable without a full architectural realignment. A `UserWarning` lists
> any missing keys at load time.

### Cache directory

All `from_pretrained` methods accept a `cache_dir` keyword:

```python
model = DepthAnything3Model.from_pretrained(
    variant="vit_l",
    cache_dir="/data/model_cache",
)
```

---

## Roadmap

### Foundation — complete

- [x] `VisionModule` base class, `Modality` / `InputForm` type system
- [x] `unicv.nn`: `DPTDecoder`, `Reassemble`, `FeatureFusionBlock` (DPT architecture)
- [x] `unicv.nn`: `SDTHead` (AnyDepth lightweight decoder)
- [x] `unicv.nn`: `MultiresConvDecoder`, `FOVNetwork` (DepthPro)
- [x] `unicv.nn`: `GaussianHead`, `GaussianCloud`, `TriangleMesh`
- [x] `unicv.nn`: `backproject_depth`, `homography_warp` (camera projection utilities)
- [x] `unicv.nn`: plane-sweep cost volume
- [x] Full pytest suite (200+ tests, mocked external downloads, offline)

### Depth estimation

| Model                                                                              | Status                                         |
| ---------------------------------------------------------------------------------- | ---------------------------------------------- |
| [DepthPro](https://github.com/apple/ml-depth-pro)                                  | **Done** — architecture + pretrained weights   |
| [Depth Anything 3](https://depth-anything-3.github.io/)                            | **Done** — architecture + pretrained weights   |
| [Camera Depth Model](https://manipulation-as-in-simulation.github.io/#cdm-results) | **Done** — architecture + pretrained weights   |
| [SimpleRecon](https://nianticlabs.github.io/simplerecon/)                          | **Done** — architecture (no public checkpoint) |

### Gaussian splat models

| Model                                                | Status                                                     |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| [SHARP](https://apple.github.io/ml-sharp/)           | **Done** — architecture + partial pretrained weights       |
| [DepthSplat](https://haofeixu.github.io/depthsplat/) | Planned — joint depth + splat from stereo/multi-view pairs |
| [InstantSplat](https://instantsplat.github.io/)      | Planned — pose-free pipeline via DUSt3R initialisation     |
| [LongSplat](https://github.com/NVlabs/LongSplat)     | Planned — online real-time splats from long video          |

### Mesh models

| Model                                                             | Status                                               |
| ----------------------------------------------------------------- | ---------------------------------------------------- |
| [SuGaR](https://anttwo.github.io/sugar/)                          | Planned — Gaussian-based surface reconstruction      |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Planned — diffusion-based single-image 3D generation |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)               | Planned — sparse 3D VAE + flow-matching transformer  |

### Point cloud models

| Model                                                  | Status                                                   |
| ------------------------------------------------------ | -------------------------------------------------------- |
| [POMATO](https://github.com/wyddmw/POMATO)             | Planned — pose-aware multi-frame RGB → point cloud       |
| [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/) | Planned — monocular SLAM via DUSt3R matching transformer |

---

## Full Model Catalogue

| Model                                                                              | Input       | Form     | Output      |
| ---------------------------------------------------------------------------------- | ----------- | -------- | ----------- |
| [DepthPro](https://github.com/apple/ml-depth-pro)                                  | RGB         | Single   | Depth       |
| [Depth Anything 3](https://depth-anything-3.github.io/)                            | RGB         | Single   | Depth       |
| [Camera Depth Model](https://manipulation-as-in-simulation.github.io/#cdm-results) | RGB + Depth | Single   | Depth       |
| [SimpleRecon](https://nianticlabs.github.io/simplerecon/)                          | RGB         | Temporal | Depth       |
| [SHARP](https://apple.github.io/ml-sharp/)                                         | RGB         | Single   | Splat       |
| [DepthSplat](https://haofeixu.github.io/depthsplat/)                               | RGB         | List     | Splat       |
| [InstantSplat](https://instantsplat.github.io/)                                    | RGB         | Temporal | Splat       |
| [LongSplat](https://github.com/NVlabs/LongSplat)                                   | RGB         | Temporal | Splat       |
| [SuGaR](https://anttwo.github.io/sugar/)                                           | RGB         | List     | Mesh        |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)                  | RGB         | Single   | Mesh        |
| [TRELLIS.2](https://microsoft.github.io/TRELLIS.2/)                                | RGB         | Single   | Mesh        |
| [POMATO](https://github.com/wyddmw/POMATO)                                         | RGB         | Temporal | Point Cloud |
| [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/)                             | RGB         | Temporal | Point Cloud |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new models, writing tests, and submitting pull requests.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to uphold a welcoming and respectful environment for everyone.

## License

MIT — see [LICENSE](LICENSE).

# UniCV

[![PyPI Latest Release](https://img.shields.io/pypi/v/unicv.svg?logo=python&logoColor=white&color=blue)](https://pypi.org/project/unicv/)
<!-- [![GitHub Release Date](https://img.shields.io/github/release-date/aether-raid/unicv?logo=github&label=latest%20release&color=blue)](https://github.com/aether-raid/unicv/releases/latest)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aether-raid/unicv/python-publish.yml?label=PyPI%20Publish&color=blue) -->

The **UniCV** library provides a unified and extensible framework for computer vision models that operate across heterogeneous input and output representations in a standard, modular manner.

The architecture and design philosophy of UniCV is inspired by modular deep learning ecosystems such as [`pytorch`](https://github.com/pytorch/pytorch) and HuggingFace's [`transformers`](https://github.com/huggingface/transformers), as well as recent efforts toward foundation models and generalist perception systems in computer vision. Rather than prescribing fixed pipelines (e.g. RGB → Depth or RGB → Mesh), UniCV abstracts vision algorithms as composable transformations between representation spaces.

At the core of UniCV lies an abstract class `VisionModule`, which defines a standardized interface for mapping *any combination of visual input modalities* to *any combination of output modalities*. These modalities include, but are not limited to:

- RGB images
- Depth maps
- Point clouds
- Meshes
- Gaussian splats and other implicit or semi-implicit scene representations

Concrete vision algorithms—such as monocular depth estimation, RGB-to-point-cloud reconstruction, or RGB-D refinement—are implemented as subclasses of this abstract interface. Existing models available online (e.g. DepthPro, MiDaS, CDM, or point-cloud reconstruction networks) can be redefined within this framework without altering their internal logic, allowing them to be seamlessly integrated into a shared system.

This abstraction enables UniCV to decouple **input modality**, **latent processing**, and **output representation**, encouraging reuse, composition, and extension of vision algorithms. Models may share encoders, latent spaces, or decoders, and can be combined or chained to support progressive or multi-stage reconstruction pipelines.

The conceptual motivation for UniCV is closely aligned with the emergence of **foundation models for perception**, where a single system is expected to reason across tasks, representations, and data sources. By enforcing a common interface at the representation level, UniCV facilitates cross-representation supervision, multi-task learning, and interoperability between otherwise incompatible vision methods.

With this design, UniCV aims to support vision systems that are:

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
unicv/
├── utils/
│   └── types.py          # Modality and InputForm enums
├── models/
│   ├── base.py           # VisionModule – the abstract interface
│   ├── depth_pro/        # DepthPro (Apple, 2024)
│   └── depth_anything_3/ # Depth Anything 3 (ByteDance, 2025)
└── nn/
    ├── decoder.py        # MultiresConvDecoder + FeatureFusionBlock2d
    ├── dpt.py            # DPTDecoder, Reassemble, FeatureFusionBlock
    ├── fov.py            # FOVNetwork (field-of-view estimation)
    └── sdt.py            # SDTHead (AnyDepth lightweight decoder)
```

### The `VisionModule` interface

Every model in UniCV inherits from `VisionModule` and declares two class attributes:

```python
from unicv.models.base import VisionModule
from unicv.utils.types import Modality, InputForm

class MyModel(VisionModule):
    input_spec = {Modality.RGB: InputForm.SINGLE}
    output_modalities = [Modality.DEPTH]

    def forward(self, **inputs):
        rgb = inputs["rgb"]   # validated and dispatched automatically
        depth = ...
        return {Modality.DEPTH: depth}
```

Calling an instance validates inputs, dispatches to `forward`, and validates outputs:

```python
model = MyModel()
result = model(rgb=image_tensor)   # → {Modality.DEPTH: tensor}
```

### Neural-network building blocks (`unicv.nn`)

| Class | Purpose |
|---|---|
| `MultiresConvDecoder` | Fuses multi-scale encoder maps (finest → coarsest) into one high-resolution feature map — used by DepthPro |
| `FeatureFusionBlock2d` | Single fusion step with optional residual skip and deconv upsampling |
| `DPTDecoder` | Full DPT pipeline: `Reassemble` patch tokens into spatial maps, then fuse with `FeatureFusionBlock` — used by DA3 |
| `Reassemble` | Converts flat ViT patch tokens to 2-D feature maps at a configurable scale |
| `FeatureFusionBlock` | DPT-style fusion with 2× bilinear upsampling |
| `FOVNetwork` | Estimates a scalar field-of-view from a low-resolution feature map |
| `SDTHead` | Lightweight SDT decoder (AnyDepth): per-level attention + depth-wise fusion |

### Implemented models

| Model | Class | Input → Output |
|---|---|---|
| [DepthPro](https://github.com/apple/ml-depth-pro) | `DepthProModel` | RGB → Depth |
| [Depth Anything 3](https://depth-anything-3.github.io/) | `DepthAnything3Model` | RGB → Depth |

---

## Roadmap

We are actively building out the core modules of **unicv**.

- [x] Set up the whole library
- [x] Implement the [DPT Architecture](https://huggingface.co/docs/transformers/v4.41.0/model_doc/dpt)
- [x] Implement the SDT Architecture from [AnyDepth](https://github.com/AIGeeksGroup/AnyDepth)
- [x] Implement [DepthPro Model](https://github.com/apple/ml-depth-pro/blob/main/src/depth_pro/depth_pro.py)
- [x] Implement [Depth Anything 3 Model](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/da3.py)

### Catalogue of models

| Name | Input | Sampling | Output |
|---|---|---|---|
| [TRELLIS.2](https://microsoft.github.io/TRELLIS.2/) | RGB | Single | Mesh |
| [Depth Anything 3 (DA3)](https://depth-anything-3.github.io/) | RGB | Single | Depth |
| [Camera Depth Model](https://manipulation-as-in-simulation.github.io/#cdm-results) | RGB + Depth | Single | Depth |
| [Depth Pro](https://github.com/apple/ml-depth-pro) | RGB | Single | Depth |
| [SHARP](https://apple.github.io/ml-sharp/) | RGB | Single | Splat |
| [SuGaR](https://anttwo.github.io/sugar/) | RGB | List | Mesh |
| [POMATO](https://github.com/wyddmw/POMATO) | RGB | Temporal | Point Cloud |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | RGB | Single | Mesh |
| [LongSplat](https://arxiv.org/abs/2507.16144) | RGB | Temporal | Splat |
| [SimpleRecon](https://nianticlabs.github.io/simplerecon/) | RGB | Temporal | Depth |
| [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/) | RGB | Temporal | Point Cloud |
| [InstantSplat](https://instantsplat.github.io/) | RGB | Temporal | Splat |
| [DepthSplat](https://haofeixu.github.io/depthsplat/) | RGB | List | Mesh, Splat |

---

## Installation and Set-Up

### Installing from PyPI

```bash
pip install unicv
```

Depending on the OS you may need `pip3` or `python -m pip install unicv`.

### Installing from source

```bash
git clone https://github.com/aether-raid/unicv.git
cd unicv
pip install .
```

To set up a full development environment with [uv](https://github.com/astral-sh/uv):

```bash
uv sync --dev
```

---

## Development

### Running the test suite

```bash
uv run pytest -v
```

Tests live in `tests/` and use [pytest](https://docs.pytest.org/). They cover all `unicv.nn` building blocks and both implemented models. `torch.hub` calls are mocked so the suite runs fully offline.

### Adding a new model

1. Create `src/unicv/models/<name>/model.py` with an `nn.Module` and a `VisionModule` wrapper.
2. Declare `input_spec` and `output_modalities` on the wrapper class.
3. Expose the public classes from `src/unicv/models/<name>/__init__.py`.
4. Re-export from `src/unicv/models/__init__.py`.
5. Add tests in `tests/test_<name>.py`.

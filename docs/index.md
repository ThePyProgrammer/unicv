# UniCV

UniCV is a unified, extensible framework for computer vision models that operate across heterogeneous input and output representations. It wraps state-of-the-art models — depth estimators, Gaussian splat predictors, mesh generators, and more — behind a single, composable `VisionModule` interface.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Install UniCV, load a pretrained model, run inference in 3 lines.

    [:octicons-arrow-right-24: Get started](getting-started.md)

-   :material-cube-outline: **VisionModule Interface**

    ---

    The core abstraction: declare modalities in, modalities out.

    [:octicons-arrow-right-24: Learn the interface](vision-module.md)

-   :material-puzzle: **Building Blocks**

    ---

    Shared decoders, heads, and geometry utilities in `unicv.nn`.

    [:octicons-arrow-right-24: Explore components](nn/overview.md)

-   :material-book-open-variant: **Model Guides**

    ---

    Per-model architecture walkthroughs with algorithmic detail.

    [:octicons-arrow-right-24: Browse models](models/catalogue.md)

</div>

---

## Quick Start

```bash
pip install unicv                  # core (torch only)
pip install unicv[pretrained]      # + huggingface_hub, timm, safetensors
```

```python
from unicv.models.depth_anything_3 import DepthAnything3Model
from unicv.utils.types import Modality

model  = DepthAnything3Model.from_pretrained(variant="vit_l")
result = model(rgb=image_tensor)
depth  = result[Modality.DEPTH]    # (B, 1, H, W)
```

Every model follows the same interface — only `input_spec` and `output_modalities` differ.

---

## Implemented Models

| Model | Paper | Input | Output | Pretrained |
|-------|-------|-------|--------|------------|
| [DepthPro](models/depth-pro.md) | Apple, 2024 | RGB | Depth | :material-check: |
| [Depth Anything 3](models/depth-anything-3.md) | ByteDance, 2025 | RGB | Depth | :material-check: |
| [Camera Depth Model](models/cdm.md) | ByteDance, 2025 | RGB + Depth | Depth | :material-check: |
| [SHARP](models/sharp.md) | Apple, 2024 | RGB | Splat | :material-check: |
| [SimpleRecon](models/simple-recon.md) | Niantic, 2022 | RGB (temporal) | Depth | :material-close: |

See the [full catalogue](models/catalogue.md) for planned models.

---

## Philosophy

Modern computer vision has fragmented into dozens of incompatible APIs: each model ships with its own preprocessing, its own output format, and its own integration burden.

The architecture and design philosophy of UniCV is inspired by modular deep learning ecosystems such as [PyTorch](https://github.com/pytorch/pytorch) and HuggingFace's [Transformers](https://github.com/huggingface/transformers), as well as recent efforts toward foundation models and generalist perception systems in computer vision. Rather than prescribing fixed pipelines (e.g. RGB → Depth or RGB → Mesh), UniCV abstracts vision algorithms as composable **transformations between representation spaces**.

The core abstraction of UniCV is `VisionModule`, which defines a standardized interface for mapping *any combination of visual input modalities* to *any combination of output modalities*. These modalities include, but are not limited to:

- RGB images
- Depth maps
- Point clouds
- Meshes
- Gaussian splats and other implicit or semi-implicit scene representations

Concrete vision algorithms — such as monocular depth estimation, RGB-to-point-cloud reconstruction, or RGB-D refinement — are implemented as subclasses of this abstract interface. Existing models available online (e.g. DepthPro, MiDaS, CDM, or point-cloud reconstruction networks) can be redefined within this framework without altering their internal logic, allowing them to be seamlessly integrated into a shared system.

This abstraction enables UniCV to decouple **input modality**, **latent processing**, and **output representation**, encouraging reuse, composition, and extension of vision algorithms. Models may share encoders, latent spaces, or decoders, and can be combined or chained to support progressive or multi-stage reconstruction pipelines.

## Motivation

UniCV aims to support vision systems that are:

1. **Modality-agnostic** — capable of ingesting arbitrary combinations of visual inputs without architectural redesign.
2. **Representation-agnostic** — able to emit multiple scene representations from a shared latent abstraction.
3. **Algorithm-agnostic** — allowing existing and future models to be wrapped, extended, or replaced under a common interface.
4. **Composable** — enabling complex pipelines by chaining or jointly training multiple `VisionModule` modules.
5. **Extensible** — supporting classical CV algorithms alongside implicit neural representations and neural rendering.
6. **Foundation-ready** — serving as an architectural substrate for generalist vision models capable of cross-task transfer.

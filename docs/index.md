# UniCV

UniCV is a unified, extensible framework for computer vision models that operate across heterogeneous input and output representations. It wraps state-of-the-art models — depth estimators, Gaussian splat predictors, mesh generators, and more — behind a single, composable `VisionModule` interface.

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

In addition to standard convolutional and Transformer-based architectures, UniCV is designed to accommodate emerging paradigms such as implicit neural representations, Gaussian splatting, and hybrid geometric-neural pipelines, enabling a unified experimental platform for next-generation 3D perception systems.

## Documentation

- [Getting Started](getting-started.md) — installation, quick usage, pretrained weights
- [VisionModule Interface](vision-module.md) — the core abstraction in detail
- [Building Blocks (`unicv.nn`)](nn/overview.md) — shared decoders, heads, and geometry utilities
- **Model Guides** — per-model architecture walkthroughs and usage:
  - [DepthPro](models/depth-pro.md)
  - [Depth Anything 3](models/depth-anything-3.md)
  - [Camera Depth Model](models/cdm.md)
  - [SHARP](models/sharp.md)
  - [SimpleRecon](models/simple-recon.md)
- [Model Catalogue](models/catalogue.md) — full table of implemented and planned models

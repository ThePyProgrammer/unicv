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

| Model | Class | Input → Output | Pretrained |
|---|---|---|---|
| [DepthPro](https://github.com/apple/ml-depth-pro) | `DepthProModel` | RGB → Depth | `DepthProModel.from_pretrained()` |
| [Depth Anything 3](https://depth-anything-3.github.io/) | `DepthAnything3Model` | RGB → Depth | `DepthAnything3Model.from_pretrained(variant=...)` |
| [Camera Depth Model](https://manipulation-as-in-simulation.github.io/#cdm-results) | `CameraDepthModel` | RGB + Depth → Depth | `CameraDepthModel.from_pretrained(camera=...)` |
| [SHARP](https://apple.github.io/ml-sharp/) | `SHARPModel` | RGB → Splat | `SHARPModel.from_pretrained()` |
| [SimpleRecon](https://nianticlabs.github.io/simplerecon/) | `SimpleReconModel` | RGB (temporal) → Depth | — |

---

## Roadmap

### Completed

- [x] Set up the library, `VisionModule` interface, `Modality`/`InputForm` type system
- [x] `unicv.nn`: `DPTDecoder`, `Reassemble`, `FeatureFusionBlock` — from the [DPT architecture](https://huggingface.co/docs/transformers/v4.41.0/model_doc/dpt)
- [x] `unicv.nn`: `SDTHead` — lightweight decoder from [AnyDepth](https://github.com/AIGeeksGroup/AnyDepth)
- [x] `unicv.nn`: `MultiresConvDecoder`, `FOVNetwork` — from [DepthPro](https://github.com/apple/ml-depth-pro)
- [x] `unicv.models.depth_pro`: `DepthProEncoder`, `DepthPro`, `DepthProModel`
- [x] `unicv.models.depth_anything_3`: `DINOv2Backbone`, `DepthAnything3`, `DepthAnything3Model`
- [x] Full pytest suite (44 tests, all `torch.hub` calls mocked, offline)

---

### Shared infrastructure

These building blocks are prerequisites for multiple catalogue models and should be implemented before the models that depend on them.

- [x] **Gaussian splat output container** — define a `GaussianCloud` dataclass (or named tensor dict) holding per-Gaussian attributes: positions `(N,3)`, scales `(N,3)`, rotations/quaternions `(N,4)`, opacities `(N,1)`, spherical-harmonic colour coefficients `(N, (deg+1)²×3)`; required by SHARP, DepthSplat, LongSplat, InstantSplat
- [x] **Gaussian parameter regression head** (`unicv.nn`) — a `GaussianHead` that maps a dense feature map to valid Gaussian parameters, using log-space scales, normalised quaternions, and sigmoid opacities; shared by SHARP and DepthSplat
- [x] **Mesh output container** — define a `TriangleMesh` dataclass holding `vertices (V,3)` and `faces (F,3)`; required by TRELLIS.2, SuGaR, Hunyuan3D-2.1
- [x] **Camera projection utilities** (`unicv.nn`) — differentiable `backproject_depth` (depth map → 3D point cloud), `homography_warp` (warp feature map between views given intrinsics/extrinsics); required by SimpleRecon, POMATO, DepthSplat
- [x] **Plane-sweep cost volume** (`unicv.nn`) — constructs a `(B, D, H, W)` matching volume by warping source frames to a set of fronto-parallel depth hypotheses using homographies, then computes NCC or learned similarity; required by SimpleRecon and DepthSplat
- [x] **3D sparse convolution wrapper** — thin wrapper around a sparse-conv library (e.g. [spconv](https://github.com/traveller59/spconv) or [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)) for voxel-grid operations; required by TRELLIS.2

---

### Depth models

- [x] **[Camera Depth Model (CDM)](https://manipulation-as-in-simulation.github.io/#cdm-results)** — dual-ViT encoder: one ViT branch for RGB, one for the raw (noisy) depth signal; tokens are fused across multiple feature levels before being fed to a `DPTDecoder` (already in `unicv.nn`). Camera-specific variants (D405, Kinect, L515) differ only in pre-trained weights. Key challenges: multi-modal token fusion at every transformer layer, sim-to-real domain gap in training data.
  - Inputs: `{Modality.RGB: SINGLE, Modality.DEPTH: SINGLE}` → `[Modality.DEPTH]`

- [x] **[SimpleRecon](https://nianticlabs.github.io/simplerecon/)** — lightweight backbone (e.g. MobileNetV2) encodes each frame independently; a plane-sweep cost volume aggregates multi-frame evidence by warping source features with homographies; 3D convolutions regularise the volume; softmax over depth candidates produces a depth map. Key challenges: cost-volume memory scales as `O(T × D × H × W)`, photometric consistency fails on textureless / reflective surfaces, requires accurate camera intrinsics.
  - Inputs: `{Modality.RGB: TEMPORAL}` → `[Modality.DEPTH]`

---

### Gaussian splat models

- [x] **[SHARP](https://apple.github.io/ml-sharp/)** — fully feed-forward (< 1 s on a standard GPU): a ViT backbone encodes the single input image; a `GaussianHead` regresses per-pixel Gaussian parameters (position, scale, rotation, opacity, SH coefficients) with metric absolute scale. Key challenges: metric scale estimation without depth supervision, defining a canonical camera-frame Gaussian layout, achieving view-consistent appearance for arbitrary novel views. (arXiv: [2512.10685](https://arxiv.org/abs/2512.10685))
  - Inputs: `{Modality.RGB: SINGLE}` → `[Modality.SPLAT]`

- [ ] **[DepthSplat](https://haofeixu.github.io/depthsplat/)** — joint depth + Gaussian prediction from a stereo or sparse multi-view pair: shared ViT/ResNet backbone; cost-volume stereo matching for geometry; a `GaussianHead` reads out splat parameters from the multi-view feature volume. Can optionally export a mesh via Poisson reconstruction on the resulting point cloud. Key challenges: view-consistent Gaussian prediction (different views may disagree on scale/rotation), handling variable numbers of input views, epipolar geometry enforcement.
  - Inputs: `{Modality.RGB: LIST}` → `[Modality.SPLAT]` (+ optionally `Modality.MESH`)

- [ ] **[InstantSplat](https://instantsplat.github.io/)** — two-stage pose-free pipeline: (1) **Coarse Geometric Initialization** — run [DUSt3R](https://github.com/naver/dust3r) (ViT + cross-attention matching transformer + 3D point-map regression head) on all image pairs to get a globally-aligned point cloud and coarse camera poses; (2) **Fast 3DGS Optimization** — seed 3D Gaussians from the DUSt3R point cloud and jointly refine Gaussian attributes + camera poses with photometric loss + pose regularisation. Key challenges: integrating or reimplementing the DUSt3R matching transformer, differentiable 3DGS rasteriser, numerical stability of joint pose + Gaussian optimisation. ([arXiv: 2403.20309](https://arxiv.org/abs/2403.20309))
  - Inputs: `{Modality.RGB: TEMPORAL}` → `[Modality.SPLAT]`

- [ ] **[LongSplat](https://github.com/NVlabs/LongSplat)** (NVIDIA, ICCV 2025) — online, real-time generalizable 3DGS from long video: per-frame Gaussians are encoded as a **Gaussian-Image Representation (GIR)** — a structured 2D image-like tensor holding all splat attributes — which enables efficient incremental update; each new frame triggers (1) **online integration** (fuse new-view Gaussians with the GIR state) and (2) **adaptive compression** (prune redundant historical Gaussians to maintain a fixed memory budget, reducing Gaussian count by ~44 %). Key challenges: designing the GIR format, handling unbounded scene growth, temporal consistency across hundreds of frames. ([arXiv: 2508.14041](https://arxiv.org/abs/2508.14041))
  - Inputs: `{Modality.RGB: TEMPORAL}` → `[Modality.SPLAT]`

---

### Mesh models

- [ ] **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** — diffusion-based single-image 3D generation: a frozen CLIP / ViT backbone conditions a 3D latent diffusion model (transformer or 3D UNet operating on tri-plane features or voxel grids); denoising produces an implicit SDF; marching cubes extracts the final mesh. Key challenges: tri-plane or voxel latent-space design, 3D UNet with spatial self-attention (expensive), marching cubes is non-differentiable (no end-to-end gradient), training requires large-scale 3D data (Objaverse-scale).
  - Inputs: `{Modality.RGB: SINGLE}` → `[Modality.MESH]`

- [ ] **[SuGaR](https://anttwo.github.io/sugar/)** (CVPR 2024) — optimization-based multi-view reconstruction: (1) run standard 3DGS on the input image collection; (2) regularise Gaussians to lie on a thin surface shell (surface-alignment loss); (3) extract a mesh via Poisson surface reconstruction on the Gaussian centres/normals. Key challenges: requires a differentiable 3DGS rasteriser (e.g. [gsplat](https://github.com/nerfstudio-project/gsplat)), Poisson reconstruction is not differentiable, full pipeline takes 30–120 min per scene, the `VisionModule` interface needs an async/lazy-forward convention to accommodate long optimization loops.
  - Inputs: `{Modality.RGB: LIST}` → `[Modality.MESH]`

- [ ] **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2)** (Microsoft, 4B params) — generates fully-textured PBR meshes at up to 1536³ voxel resolution via a **Sparse Compression 3D VAE** (16× spatial downsampling, sparse residual autoencoder) and a **flow-matching transformer** operating on O-Voxel structured latents; bidirectional O-Voxel ↔ mesh conversion is seconds-fast on CPU. Key challenges: requires custom CUDA/Triton kernels (FlexGEMM for sparse convolutions, CuMesh for remeshing/UV unwrapping), O-Voxel is a non-standard 3D representation with a dedicated dual-grid geometry + PBR material encoding, 4B-parameter model weight management.
  - Inputs: `{Modality.RGB: SINGLE}` → `[Modality.MESH]`

---

### Point cloud models

- [ ] **[POMATO](https://github.com/wyddmw/POMATO)** — pose-aware multi-frame RGB → point cloud: a shared CNN / ViT backbone extracts per-frame features; an optical-flow estimator (RAFT-style) warps features between frames; a temporal attention module aggregates evidence across the sequence; a depth / 3D regression head predicts per-pixel 3D coordinates. Key challenges: optical-flow errors accumulate in long sequences, 4D cost-volume memory `O(T × D × H × W)` is large, occlusion handling, camera-pose accuracy strongly affects reconstruction quality.
  - Inputs: `{Modality.RGB: TEMPORAL}` → `[Modality.POINT_CLOUD]`

- [ ] **[MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/)** — full monocular SLAM system built on [DUSt3R](https://github.com/naver/dust3r): a ViT + cross-attention **matching transformer** predicts dense correspondences and per-pixel 3D point-map positions for every image pair; a sliding-window pose graph accumulates keyframes; bundle adjustment (differentiable or classical Levenberg–Marquardt) refines camera trajectories; loop-closure re-runs the matching transformer on stored keyframe pairs. Key challenges: the matching transformer is quadratic in image pairs `O(K²)`, differentiable bundle adjustment is non-standard (requires sparse linear-solve backprop or a neural approximation), dynamic scene objects violate static-scene assumptions, metric scale is inherently ambiguous without a depth sensor.
  - Inputs: `{Modality.RGB: TEMPORAL}` → `[Modality.POINT_CLOUD]`

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

## Loading pretrained models

Each implemented model ships with a `from_pretrained` classmethod that downloads the official checkpoint from Hugging Face Hub and loads it into the UniCV architecture.

Install the optional dependencies first:

```bash
pip install huggingface_hub timm safetensors
```

---

### DepthPro

Loads Apple's DepthPro weights from [`apple/DepthPro`](https://huggingface.co/apple/DepthPro).  Requires `huggingface_hub` and `timm`.

```python
from unicv.models.depth_pro import DepthProModel

model = DepthProModel.from_pretrained()                        # default: includes FoV head
model = DepthProModel.from_pretrained(use_fov_head=False)      # depth-only variant
model.eval()

result = model(rgb=image_tensor)   # image_tensor: (B, 3, 1536, 1536)
depth  = result["depth"]           # metric depth in metres, same spatial size
```

---

### Depth Anything 3

Downloads checkpoints from the [`depth-anything`](https://huggingface.co/depth-anything) organisation.  Four backbone sizes are available.  Requires `huggingface_hub` and `safetensors`.

| `variant` | Hugging Face repo | Backbone embed dim |
|---|---|---|
| `"vit_s"` | `depth-anything/DA3-SMALL`  | 384  |
| `"vit_b"` | `depth-anything/DA3-BASE`   | 768  |
| `"vit_l"` | `depth-anything/DA3-LARGE` *(default)* | 1024 |
| `"vit_g"` | `depth-anything/DA3-GIANT`  | 1536 |

```python
from unicv.models.depth_anything_3 import DepthAnything3Model

model = DepthAnything3Model.from_pretrained(variant="vit_l")   # default
model.eval()

result = model(rgb=image_tensor)   # image_tensor: (B, 3, H, W)
depth  = result["depth"]           # inverse-depth map, shape (B, 1, H, W)
```

---

### Camera Depth Model (CDM)

Downloads camera-specific checkpoints from [`depth-anything/camera-depth-model-{camera}`](https://huggingface.co/depth-anything).  Choose the variant that matches your depth sensor.  Requires `huggingface_hub`.

| `camera` | Hugging Face repo | Sensor |
|---|---|---|
| `"d405"` *(default)* | `depth-anything/camera-depth-model-d405`   | Intel RealSense D405 |
| `"d435"` | `depth-anything/camera-depth-model-d435`   | Intel RealSense D435 |
| `"l515"` | `depth-anything/camera-depth-model-l515`   | Intel RealSense L515 |
| `"kinect"` | `depth-anything/camera-depth-model-kinect` | Azure Kinect |

```python
from unicv.models.cdm import CameraDepthModel

model = CameraDepthModel.from_pretrained(camera="d405")   # default
model.eval()

result  = model(rgb=rgb_tensor, depth=raw_depth_tensor)   # both (B, *, H, W)
refined = result["depth"]                                  # refined depth, (B, 1, H, W)
```

---

### SHARP

Downloads the SHARP checkpoint directly from Apple's CDN (no Hugging Face dependency).  No extra dependencies beyond PyTorch.

```python
from unicv.models.sharp import SHARPModel

model = SHARPModel.from_pretrained()
model.eval()

result = model(rgb=image_tensor)   # image_tensor: (B, 3, H, W)
cloud  = result["splat"]           # GaussianCloud with N = H×W Gaussians
```

> **Architecture note** — the official SHARP checkpoint stores an `RGBGaussianPredictor`
> whose encoder is DepthPro-based (`SlidingPyramidNetwork` + `TimmViT`), not DINOv2.
> The following partial remappings are applied where shapes align:
> fusion-block conv weights from `gaussian_decoder.fusions.*` load into
> `feature_decoder.fusion_blocks.*`, and the geometry prediction head loads into
> `gaussian_head.xyz_head` when the output channel count matches.
> Backbone and reassemble-block weights require an architectural realignment with
> the official implementation for a full load.  A `UserWarning` lists missing keys.

---

### Cache directory

All three methods accept a `cache_dir` keyword that is forwarded directly to `hf_hub_download`.  When omitted, weights land in the default Hugging Face cache (`~/.cache/huggingface`).

```python
model = DepthAnything3Model.from_pretrained(
    variant="vit_l",
    cache_dir="/data/hf_cache",
)
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

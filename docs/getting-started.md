# Getting Started

## Installation

### From PyPI

```bash
pip install unicv
```

This installs the core library with only `torch` as a dependency.

### Pretrained weights

To use `from_pretrained()` on any model, install the optional extras:

```bash
pip install unicv[pretrained]
```

This adds `huggingface_hub`, `timm`, and `safetensors`.

### From source

```bash
git clone https://github.com/ThePyProgrammer/unicv.git
cd unicv
pip install .
```

### Development

Requires [uv](https://github.com/astral-sh/uv):

```bash
uv sync --dev
uv run pytest -v   # 200+ tests, fully offline
```

## Quick Usage

Every model in UniCV follows the same pattern:

```python
# 1. Import the VisionModule wrapper
from unicv.models.depth_anything_3 import DepthAnything3Model

# 2. Load pretrained weights
model = DepthAnything3Model.from_pretrained(variant="vit_l")
model.eval()

# 3. Call with named modality inputs
result = model(rgb=image_tensor)     # (B, 3, H, W)
depth  = result[Modality.DEPTH]      # (B, 1, H, W)
```

The interface is identical for all models. Only the `input_spec` and
`output_modalities` differ:

```python
from unicv.models.sharp import SHARPModel

model  = SHARPModel.from_pretrained()
result = model(rgb=image_tensor)
cloud  = result[Modality.SPLAT]      # GaussianCloud
```

## Loading Pretrained Weights

### DepthPro

```python
from unicv.models.depth_pro import DepthProModel

model = DepthProModel.from_pretrained()               # includes FOV head
model = DepthProModel.from_pretrained(use_fov_head=False)
```

Downloads from [`apple/DepthPro`](https://huggingface.co/apple/DepthPro) on Hugging Face.

### Depth Anything 3

```python
from unicv.models.depth_anything_3 import DepthAnything3Model

model = DepthAnything3Model.from_pretrained(variant="vit_l")
```

Available variants: `"vit_s"` (384), `"vit_b"` (768), `"vit_l"` (1024), `"vit_g"` (1536).

### Camera Depth Model

```python
from unicv.models.cdm import CameraDepthModel

model = CameraDepthModel.from_pretrained(camera="d405")
result = model(rgb=rgb_tensor, depth=raw_depth_tensor)
```

Camera variants: `"d405"`, `"d435"`, `"l515"`, `"kinect"`.

### SHARP

```python
from unicv.models.sharp import SHARPModel

model = SHARPModel.from_pretrained()
cloud = model(rgb=image_tensor)[Modality.SPLAT]
```

Downloads from Apple's CDN (no Hugging Face dependency).

> **Note:** The official SHARP checkpoint uses a DepthPro-based encoder, not
> DINOv2. UniCV applies partial weight remappings where shapes align.
> A `UserWarning` lists any missing keys.

### Cache directory

All `from_pretrained` methods accept `cache_dir`:

```python
model = DepthAnything3Model.from_pretrained(
    variant="vit_l",
    cache_dir="/data/model_cache",
)
```

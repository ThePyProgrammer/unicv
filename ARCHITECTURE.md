# Architecture

This document describes the high-level architecture of UniCV.
If you want to familiarize yourself with the codebase, you are in the right place.

## Bird's Eye View

UniCV wraps published computer vision models behind a single `VisionModule`
interface so that different modalities (RGB, depth, point clouds, Gaussian
splats, meshes) flow in and out through a uniform contract. The library is
pure PyTorch with no framework dependencies beyond `torch` itself.

Every model follows a **two-layer pattern**:

1. `Foo(nn.Module)` -- the real architecture, standard PyTorch, no unicv
   abstractions. You can use it standalone.
2. `FooModel(VisionModule)` -- a thin wrapper that plugs `Foo` into the
   unicv type system. It validates inputs against `input_spec`, calls
   `forward`, and validates outputs against `output_modalities`.

This means the library never forces you into its abstractions; the inner
`nn.Module` is always accessible for direct use.

## Code Map

### `src/unicv/utils/` -- leaf types, no dependencies

The bottom of the dependency graph. Nothing else in `unicv` is imported here.

- `types.py`: `Modality` and `InputForm` enums. These two enums are the
  vocabulary that every model uses to declare what it consumes and produces.
- `structs.py`: `GaussianCloud` and `TriangleMesh` dataclasses. Structured
  containers for non-tensor outputs (splats, meshes).

### `src/unicv/nn/` -- reusable building blocks

Neural network modules shared across multiple models. Depends on `utils/`
only. Nothing here knows about `VisionModule` or any specific model.

- `dinov2.py`: `DINOv2Backbone` -- wraps `torch.hub` DINOv2 models and
  captures intermediate hidden states via forward hooks. Shared by
  DepthAnything3, CDM, and SHARP.
- `dpt.py`: `DPTDecoder`, `Reassemble`, `FeatureFusionBlock` -- the Dense
  Prediction Transformer decoder that reassembles ViT tokens into spatial
  feature maps. The workhorse decoder for most models.
- `decoder.py`: `MultiresConvDecoder` -- an alternative multi-resolution
  convolutional decoder used by DepthPro.
- `sdt.py`: `SDTHead` -- lightweight cross-scale attention decoder.
- `fov.py`: `FOVNetwork` -- field-of-view estimation head for DepthPro.
- `gaussian.py`: `GaussianHead` -- regresses Gaussian splat parameters from
  a dense feature map.
- `geometry.py`: `backproject_depth`, `homography_warp`, `default_intrinsics`
  -- differentiable camera geometry primitives.
- `cost_volume.py`: `PlaneSweepCostVolume` -- builds a matching volume from
  multi-view features at discrete depth hypotheses.
- `sparse3d.py`: `SparseConv3d` -- optional sparse 3D convolution with
  backend auto-detection (spconv or MinkowskiEngine).

### `src/unicv/models/` -- concrete model implementations

Each model lives in its own sub-package (`depth_pro/`, `depth_anything_3/`,
`cdm/`, `sharp/`, `simple_recon/`) containing a `model.py` and an
`__init__.py` that re-exports public names.

- `base.py`: `VisionModule` abstract base class. Defines the `input_spec` /
  `output_modalities` contract and handles input parsing, output validation,
  and dispatch to `forward`. Also contains shared checkpoint-loading helpers
  (`_require_package`, `_warn_missing_keys`, `_remap_dpt_key`).
- `depth_pro/`: DepthPro -- multi-scale patch-pyramid encoder with a custom
  `DepthProEncoder`, multi-resolution conv decoder, and optional FOV head.
- `depth_anything_3/`: Depth Anything 3 -- DINOv2 backbone + DPT decoder.
- `cdm/`: Camera Depth Model -- dual-ViT (RGB + depth) with per-level token
  fusion, decoded by DPT.
- `sharp/`: SHARP -- single-image Gaussian splat prediction via DINOv2 +
  DPT + GaussianHead + depth backprojection.
- `simple_recon/`: SimpleRecon -- multi-view plane-sweep stereo with a
  lightweight CNN encoder, cost volume, and 3D regularizer.

### `tests/`

One test file per source module, plus `test_pretrained.py` which tests
`from_pretrained()` for all models with mocked downloads. Tests use small
tensors and never make network calls.

## Dependency Flow

```
utils/        (Modality, InputForm, GaussianCloud, TriangleMesh)
  ^
  |
nn/           (DINOv2Backbone, DPTDecoder, GaussianHead, geometry, ...)
  ^
  |
models/       (VisionModule, DepthPro, DA3, CDM, SHARP, SimpleRecon)
```

Dependencies are strictly one-way: `models` imports from `nn` and `utils`;
`nn` imports from `utils`; `utils` imports nothing from unicv. There are no
import cycles.

## Cross-Cutting Concerns

**Optional dependencies.** Only `torch` is required at install time.
Pretrained weight loading needs `huggingface_hub`, `timm`, and/or
`safetensors`, but these are gated behind lazy `_require_package()` checks
and only installed via `pip install unicv[pretrained]`.

**Checkpoint key remapping.** Official checkpoints use different naming
conventions than unicv's module structure. Each `from_pretrained` method
remaps keys before calling `load_state_dict(strict=False)`. Common DPT
decoder remapping is shared via `_remap_dpt_key` in `base.py`.

**The `from_config` / `from_pretrained` split.** Every model's `nn.Module`
has a `from_config()` classmethod that builds the architecture from
high-level parameters. The `VisionModule` wrapper's `from_pretrained()`
calls `from_config()` then loads weights. This keeps architecture
construction separate from weight loading.

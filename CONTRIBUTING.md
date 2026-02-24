# Contributing to UniCV

Thank you for your interest in contributing! This guide covers everything you need to get started — from setting up the development environment to submitting a pull request.

---

## Table of Contents

- [Development setup](#development-setup)
- [Running the tests](#running-the-tests)
- [Project structure](#project-structure)
- [Adding a new model](#adding-a-new-model)
- [Adding a `unicv.nn` building block](#adding-a-unicvnn-building-block)
- [Code style](#code-style)
- [Submitting a pull request](#submitting-a-pull-request)
- [Reporting issues](#reporting-issues)

---

## Development setup

UniCV uses [uv](https://github.com/astral-sh/uv) as its package manager. Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or on Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex
```

Then clone the repository and install all dependencies (including dev extras):

```bash
git clone https://github.com/aether-raid/unicv.git
cd unicv
uv sync --dev
```

Always use `uv run <cmd>` to execute commands inside the project's managed environment.

---

## Running the tests

```bash
uv run pytest -v
```

The full suite should pass in around 30 seconds. All external downloads (`torch.hub`, `hf_hub_download`, etc.) are mocked, so the suite runs fully offline. CI runs the same command across Ubuntu, macOS, and Windows on Python 3.10–3.13.

To run a specific test file:

```bash
uv run pytest tests/test_dpt.py -v
```

---

## Project structure

```
src/unicv/
├── utils/types.py          # Modality and InputForm enums — the only types file
├── models/
│   ├── base.py             # VisionModule abstract base class
│   └── <name>/
│       ├── model.py        # Foo(nn.Module) + FooModel(VisionModule)
│       └── __init__.py
└── nn/
    ├── decoder.py          # MultiresConvDecoder, FeatureFusionBlock2d
    ├── dpt.py              # DPTDecoder, Reassemble, FeatureFusionBlock
    ├── fov.py              # FOVNetwork
    ├── sdt.py              # SDTHead
    ├── gaussian.py         # GaussianHead
    └── geometry.py         # backproject_depth, homography_warp
tests/
    test_<name>.py          # one file per module / model
```

Every model follows a two-class pattern:

1. `Foo(nn.Module)` — pure PyTorch, owns the architecture, has a standard `forward(x)`.
2. `FooModel(VisionModule)` — thin wrapper that plugs `Foo` into the UniCV interface.

---

## Adding a new model

Follow these steps to integrate a new model:

### 1. Create the model files

```
src/unicv/models/<name>/
    __init__.py    # re-export public classes
    model.py       # Foo(nn.Module) + FooModel(VisionModule)
```

`model.py` skeleton:

```python
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from unicv.models.base import VisionModule
from unicv.utils.types import InputForm, Modality


class Foo(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class FooModel(VisionModule):
    input_spec        = {Modality.RGB: InputForm.SINGLE}
    output_modalities = [Modality.DEPTH]

    def __init__(self, net: Foo) -> None:
        super().__init__()
        self.net = net

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        rgb = inputs[Modality.RGB.value]
        return {Modality.DEPTH: self.net(rgb)}

    @classmethod
    def from_pretrained(cls, ...) -> "FooModel":
        ...
```

### 2. Register the model

In `src/unicv/models/__init__.py`, add imports and update `__all__`:

```python
from unicv.models.<name>.model import Foo, FooModel
__all__ = [..., "Foo", "FooModel"]
```

### 3. Write tests

Create `tests/test_<name>.py`. At minimum cover:

- Instantiation succeeds without errors.
- `forward()` returns the expected output shape.
- `VisionModule` validation works (wrong modality raises, correct modality passes).
- `from_pretrained()` loads without error (mock all external downloads).

```python
from unittest.mock import patch
import torch
from unicv.models.<name>.model import Foo, FooModel
from unicv.utils.types import Modality

def test_forward_shape():
    model = FooModel(net=Foo(...))
    rgb   = torch.randn(2, 3, 64, 64)
    out   = model(rgb=rgb)
    assert out[Modality.DEPTH].shape == (2, 1, 64, 64)

def test_from_pretrained():
    with patch("huggingface_hub.hf_hub_download", return_value="/tmp/fake.pt"), \
         patch("torch.load", return_value={}):
        model = FooModel.from_pretrained()
    assert isinstance(model, FooModel)
```

### 4. Update the README

Add a row to the "Implemented Models" table in `README.md` and, if the model has a `from_pretrained`, add a usage example under "Loading Pretrained Weights".

### 5. Verify

```bash
uv run pytest -v
```

All tests must pass before opening a pull request.

---

## Adding a `unicv.nn` building block

New `nn` modules live in `src/unicv/nn/`. Follow the same shape-test convention as existing modules (see `tests/test_dpt.py` for reference). Export the class from `src/unicv/nn/__init__.py` and add a corresponding `tests/test_<module>.py`.

---

## Code style

- **Python 3.10+** — use `X | Y` union syntax rather than `Optional[X]` or `Union[X, Y]`.
- **No numpy, no torchvision** — only `torch`, `einops`, `timm`, and `huggingface_hub` are core dependencies.
- **No over-engineering** — do not add abstractions, helpers, or error handling for scenarios that do not exist yet. Three similar lines of code is better than a premature abstraction.
- **No docstrings on trivial methods** — only add docstrings where the logic or contract is non-obvious.
- **Type annotations** on all public functions and class attributes.
- Keep lines under 120 characters where practical.

---

## Submitting a pull request

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/my-model
   ```
2. Make your changes and run the full test suite:
   ```bash
   uv run pytest -v
   ```
3. Open a pull request against `main`. The PR description should explain:
   - What the change does.
   - Which model or building block is added / changed.
   - Any architectural decisions or trade-offs.
4. CI will run the test matrix automatically. Address any failures before requesting review.

---

## Reporting issues

Please use the [GitHub issue tracker](https://github.com/aether-raid/unicv/issues). Include:

- Your Python version (`uv run python --version`).
- Your PyTorch version (`uv run python -c "import torch; print(torch.__version__)"`).
- A minimal reproducible example.

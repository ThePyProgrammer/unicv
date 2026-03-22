# The VisionModule Interface

`VisionModule` is the core abstraction in UniCV. Every model inherits from it, declaring what modalities it consumes and produces.

## Contract

A `VisionModule` subclass must define two class attributes and one method:

```python
class MyModel(VisionModule):
    input_spec: dict[Modality, InputForm] = {
        Modality.RGB: InputForm.SINGLE,
    }
    output_modalities: list[Modality] = [Modality.DEPTH]

    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        rgb = inputs["rgb"]
        depth = self.net(rgb)
        return {Modality.DEPTH: depth}
```

### `Modality`

An enum of data representations that flow between models:

| Value          | Typical tensor shape       |
|----------------|----------------------------|
| `RGB`          | `(B, 3, H, W)`            |
| `DEPTH`        | `(B, 1, H, W)`            |
| `POINT_CLOUD`  | `(B, N, 3)`               |
| `MESH`         | `TriangleMesh` dataclass   |
| `SPLAT`        | `GaussianCloud` dataclass  |
| `LATENT`       | model-dependent            |

### `InputForm`

Declares the shape of each input modality:

| Value      | Meaning                              |
|------------|--------------------------------------|
| `SINGLE`   | One tensor or object                 |
| `LIST`     | Unordered collection (multi-view)    |
| `TEMPORAL`  | Ordered time sequence (video)       |

## Call flow

When you call a `VisionModule` instance:

```
model(rgb=tensor, depth=tensor)
  │
  ├─ _parse_inputs()     # validate against input_spec
  │   ├─ check all required modalities are present
  │   ├─ SINGLE: pass through
  │   └─ LIST/TEMPORAL: verify list/tuple type
  │
  ├─ forward(**parsed)   # your model logic
  │
  └─ _validate_outputs() # check keys against output_modalities
```

Input keys are the **string values** of `Modality` members (e.g. `"rgb"`, `"depth"`). The `forward` method receives them as keyword arguments.

## The two-layer pattern

Every implemented model follows a two-layer split:

1. **`Foo(nn.Module)`** — pure PyTorch. Standard `forward(x)` signature. No awareness of `Modality` or `InputForm`. Can be used standalone, tested independently, or plugged into any PyTorch pipeline.

2. **`FooModel(VisionModule)`** — thin wrapper. Holds a `self.net` reference to the `nn.Module`, translates between the modality contract and the raw tensor interface.

This means:
- You can always drop down to the raw module: `model.net(tensor)`
- The `VisionModule` layer adds validation, not complexity
- Pretrained weights go to the inner module via `from_pretrained()`

## Validation behaviour

**At construction:** `__init__` calls `_validate_class_definition()` which checks that `input_spec` is non-empty and all keys/values are valid `Modality`/`InputForm` members.

**At call time:** Missing inputs raise `KeyError`. Wrong types for `LIST`/`TEMPORAL` inputs raise `TypeError`. Output modalities not declared in `output_modalities` raise `ValueError`.

## Output structs

For non-tensor outputs, UniCV provides typed dataclasses:

- **`GaussianCloud`** — `xyz`, `scales`, `rotations`, `opacities`, `sh_coeffs`. Used by `Modality.SPLAT`.
- **`TriangleMesh`** — `vertices`, `faces`. Used by `Modality.MESH`.

Both support `.to(device)` and `.detach()` for device transfer and gradient detaching across all contained tensors.

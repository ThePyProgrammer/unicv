"""VisionModule base class and shared checkpoint-loading utilities."""

from abc import ABC, abstractmethod
from typing import Any
import warnings

from unicv.utils.types import Modality, InputForm


def _require_package(name: str, pip_name: str | None = None) -> None:
    """Import *name* or raise ImportError with install instructions."""
    import importlib
    try:
        importlib.import_module(name)
    except ImportError as e:
        install = pip_name or name
        raise ImportError(
            f"This operation requires {name}.\n"
            f"Install it with:  pip install {install}"
        ) from e


def _remap_dpt_key(key: str) -> str | None:
    """Remap a single official DPT checkpoint key to unicv naming.

    Returns the remapped key, or ``None`` if the key does not match any known
    DPT pattern and should be handled by the caller.
    """
    if key.startswith("depth_head.projects."):
        rest = key[len("depth_head.projects."):]
        idx, _, tail = rest.partition(".")
        return f"decoder.reassemble_blocks.{idx}.project.{tail}"

    if key.startswith("depth_head.resize_layers."):
        rest = key[len("depth_head.resize_layers."):]
        idx, _, tail = rest.partition(".")
        return f"decoder.reassemble_blocks.{idx}.resample.{tail}"

    if key.startswith("depth_head.scratch.refinenet"):
        rest = key[len("depth_head.scratch.refinenet"):]
        n_str, _, tail = rest.partition(".")
        i = 4 - int(n_str)
        tail = tail.replace("resConfUnit", "res_conv_unit")
        return f"decoder.fusion_blocks.{i}.{tail}"

    if key.startswith("depth_head.scratch.output_conv1."):
        tail = key[len("depth_head.scratch.output_conv1."):]
        return f"decoder.head.0.{tail}"

    if key.startswith("depth_head.scratch.output_conv2."):
        tail = key[len("depth_head.scratch.output_conv2."):]
        return f"decoder.head.2.{tail}"

    return None


def _warn_missing_keys(
    model_label: str, missing: list[str], *, limit: int = 5
) -> None:
    """Emit a UserWarning listing the first *limit* missing state-dict keys."""
    if missing:
        shown = missing[:limit]
        warnings.warn(
            f"{model_label}: {len(missing)} missing key(s) when loading "
            f"pretrained weights (showing {len(shown)}): {shown}",
            stacklevel=3,
        )


class VisionModule(ABC):
    """Abstract base class for any-to-any vision algorithms.

    Subclasses define:
    - ``input_spec: dict[Modality, InputForm]``
    - ``output_modalities: list[Modality]``
    """

    input_spec: dict[Modality, InputForm] = {}
    output_modalities: list[Modality] = []

    def __init__(self):
        self._validate_class_definition()

    def _validate_class_definition(self):
        if not self.input_spec:
            raise ValueError(
                f"{self.__class__.__name__} must define `input_spec`"
            )

        for modality, form in self.input_spec.items():
            if not isinstance(modality, Modality):
                raise TypeError(f"Invalid modality key: {modality}")
            if not isinstance(form, InputForm):
                raise TypeError(f"Invalid input form for {modality}: {form}")

        for modality in self.output_modalities:
            if not isinstance(modality, Modality):
                raise TypeError(f"Invalid output modality: {modality}")

    def __call__(self, **kwargs) -> dict[Modality, Any]:
        """
        Parse kwargs according to input_spec and dispatch to forward().
        """
        parsed_inputs = self._parse_inputs(kwargs)
        outputs = self.forward(**parsed_inputs)
        return self._validate_outputs(outputs)

    def _parse_inputs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}

        for modality, form in self.input_spec.items():
            key = modality.value

            if key not in kwargs:
                raise KeyError(
                    f"Missing required input modality '{key}' "
                    f"for {self.__class__.__name__}"
                )

            value = kwargs[key]

            if form == InputForm.SINGLE:
                parsed[key] = value

            elif form in (InputForm.LIST, InputForm.TEMPORAL):
                if not isinstance(value, (list, tuple)):
                    raise TypeError(
                        f"Expected {form.value} (list/tuple) for '{key}'"
                    )
                parsed[key] = list(value)

            else:
                raise RuntimeError(f"Unhandled InputForm: {form}")

        return parsed

    def _validate_outputs(
        self, outputs: dict[Modality, Any]
    ) -> dict[Modality, Any]:

        if not isinstance(outputs, dict):
            raise TypeError(
                "forward() must return dict[Modality, Any]"
            )

        normalized: dict[Modality, Any] = {}

        for key, value in outputs.items():
            modality = (
                key if isinstance(key, Modality) else Modality(key)
            )

            if modality not in self.output_modalities:
                raise ValueError(
                    f"Unexpected output modality '{modality}' "
                    f"from {self.__class__.__name__}"
                )

            normalized[modality] = value

        return normalized

    @abstractmethod
    def forward(self, **inputs: Any) -> dict[Modality, Any]:
        """Subclasses implement actual logic here.

        Inputs are already validated and normalized.
        """
        pass


__all__ = [
    "VisionModule",
    "_remap_dpt_key",
    "_require_package",
    "_warn_missing_keys",
]

"""VisionModule abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from unicv.utils.types import Modality, InputForm

# Re-export checkpoint helpers for backwards compatibility.
from unicv.models._checkpoint import (  # noqa: F401
    _remap_checkpoint,
    _remap_dpt_key,
    _require_package,
    _warn_missing_keys,
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
        """Parse kwargs according to input_spec and dispatch to forward()."""
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
    "_remap_checkpoint",
    "_remap_dpt_key",
    "_require_package",
    "_warn_missing_keys",
]

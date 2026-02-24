"""Tests for VisionModule base class (unicv.models.base)."""

import pytest
import torch

from unicv.models.base import VisionModule
from unicv.utils.types import InputForm, Modality


# ---------------------------------------------------------------------------
# Concrete dummy subclass
# ---------------------------------------------------------------------------

class _SingleInputModule(VisionModule):
    """A minimal VisionModule that passes rgb through as depth."""

    input_spec = {Modality.RGB: InputForm.SINGLE}
    output_modalities = [Modality.DEPTH]

    def forward(self, **inputs):
        return {Modality.DEPTH: inputs["rgb"]}


class _ListInputModule(VisionModule):
    """A minimal VisionModule that accepts a list of RGB frames."""

    input_spec = {Modality.RGB: InputForm.LIST}
    output_modalities = [Modality.DEPTH]

    def forward(self, **inputs):
        return {Modality.DEPTH: inputs["rgb"][0]}


# ---------------------------------------------------------------------------
# Instantiation / class-definition validation
# ---------------------------------------------------------------------------

def test_valid_module_instantiation():
    m = _SingleInputModule()
    assert isinstance(m, VisionModule)


def test_module_without_input_spec_raises():
    class _BadModule(VisionModule):
        input_spec = {}
        output_modalities = [Modality.DEPTH]

        def forward(self, **inputs):
            return {Modality.DEPTH: None}

    with pytest.raises(ValueError, match="must define `input_spec`"):
        _BadModule()


def test_invalid_modality_key_raises():
    class _BadKeys(VisionModule):
        input_spec = {"not_a_modality": InputForm.SINGLE}
        output_modalities = [Modality.DEPTH]

        def forward(self, **inputs):
            return {Modality.DEPTH: None}

    with pytest.raises(TypeError, match="Invalid modality key"):
        _BadKeys()


# ---------------------------------------------------------------------------
# __call__ dispatching
# ---------------------------------------------------------------------------

def test_call_dispatches_correctly():
    m = _SingleInputModule()
    x = torch.zeros(1, 3, 4, 4)
    result = m(rgb=x)
    assert Modality.DEPTH in result
    assert torch.equal(result[Modality.DEPTH], x)


def test_missing_required_input_raises():
    m = _SingleInputModule()
    with pytest.raises(KeyError):
        m(depth=torch.zeros(1, 1, 4, 4))  # "rgb" key is missing


def test_list_input_form_accepted():
    m = _ListInputModule()
    frames = [torch.zeros(1, 3, 4, 4) for _ in range(3)]
    result = m(rgb=frames)
    assert Modality.DEPTH in result


def test_list_input_form_rejects_scalar():
    m = _ListInputModule()
    with pytest.raises(TypeError, match="Expected list/tuple"):
        m(rgb=torch.zeros(1, 3, 4, 4))  # should be a list


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def test_unexpected_output_modality_raises():
    class _BadOutput(VisionModule):
        input_spec = {Modality.RGB: InputForm.SINGLE}
        output_modalities = [Modality.DEPTH]

        def forward(self, **inputs):
            # Returns an undeclared output modality.
            return {Modality.POINT_CLOUD: inputs["rgb"]}

    m = _BadOutput()
    with pytest.raises(ValueError, match="Unexpected output modality"):
        m(rgb=torch.zeros(1, 3, 4, 4))

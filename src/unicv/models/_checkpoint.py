"""Shared checkpoint-loading utilities for from_pretrained methods."""

from __future__ import annotations

import warnings

import torch


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


def _remap_checkpoint(
    raw_sd: dict[str, torch.Tensor],
    prefix_map: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Remap a checkpoint state dict using prefix substitutions + DPT keys.

    Args:
        raw_sd: Raw state dict from the checkpoint.
        prefix_map: Mapping of source prefixes to target prefixes,
            e.g. ``{"pretrained.": "backbone.model."}``.

    Returns:
        Remapped state dict.  Unrecognised keys pass through unchanged.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, val in raw_sd.items():
        new_key = key

        # Try prefix substitutions first.
        for src_prefix, dst_prefix in prefix_map.items():
            if new_key.startswith(src_prefix):
                new_key = dst_prefix + new_key[len(src_prefix):]
                break
        else:
            # No prefix matched -- try DPT key remapping.
            dpt_key = _remap_dpt_key(new_key)
            if dpt_key is not None:
                new_key = dpt_key

        remapped[new_key] = val
    return remapped


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


__all__ = [
    "_remap_checkpoint",
    "_remap_dpt_key",
    "_require_package",
    "_warn_missing_keys",
]

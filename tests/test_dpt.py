"""Tests for DPTDecoder, FeatureFusionBlock, and Reassemble (unicv.nn.dpt)."""

import pytest
import torch

from unicv.nn.dpt import DPTDecoder, FeatureFusionBlock, Reassemble


# ---------------------------------------------------------------------------
# Shared constants (small values for speed)
# ---------------------------------------------------------------------------

EMBED_DIM = 256
PATCH_SIZE = 16
IMG_SIZE = 64        # → h_base = 64 // 16 = 4  (4×4 patch grid)
FEATURES = 64
B = 2


def _make_tokens(num_register_tokens: int = 0) -> torch.Tensor:
    """Return a fake ViT token sequence (B, 1 + reg + N, D)."""
    N = (IMG_SIZE // PATCH_SIZE) ** 2  # 16 patch tokens
    seq_len = 1 + num_register_tokens + N
    return torch.zeros(B, seq_len, EMBED_DIM)


# ---------------------------------------------------------------------------
# Reassemble
# ---------------------------------------------------------------------------

def test_reassemble_scale_1():
    """Scale=1 keeps the native patch-grid resolution."""
    r = Reassemble(EMBED_DIM, FEATURES, PATCH_SIZE, IMG_SIZE, layer_scale=1)
    out = r(_make_tokens())
    h = IMG_SIZE // PATCH_SIZE  # 4
    assert out.shape == (B, FEATURES, h, h)


def test_reassemble_scale_2():
    """Scale=2 doubles the spatial size via ConvTranspose2d."""
    r = Reassemble(EMBED_DIM, FEATURES, PATCH_SIZE, IMG_SIZE, layer_scale=2)
    out = r(_make_tokens())
    h = (IMG_SIZE // PATCH_SIZE) * 2  # 8
    assert out.shape == (B, FEATURES, h, h)


def test_reassemble_scale_4():
    """Scale=4 quadruples the spatial size via two ConvTranspose2d."""
    r = Reassemble(EMBED_DIM, FEATURES, PATCH_SIZE, IMG_SIZE, layer_scale=4)
    out = r(_make_tokens())
    h = (IMG_SIZE // PATCH_SIZE) * 4  # 16
    assert out.shape == (B, FEATURES, h, h)


def test_reassemble_scale_0_5():
    """Scale=0.5 halves the spatial size via a strided Conv2d."""
    r = Reassemble(EMBED_DIM, FEATURES, PATCH_SIZE, IMG_SIZE, layer_scale=0.5)
    out = r(_make_tokens())
    h = (IMG_SIZE // PATCH_SIZE) // 2  # 2
    assert out.shape == (B, FEATURES, h, h)


def test_reassemble_with_register_tokens():
    """Register tokens are dropped before reshaping."""
    r = Reassemble(EMBED_DIM, FEATURES, PATCH_SIZE, IMG_SIZE, layer_scale=1, num_register_tokens=4)
    out = r(_make_tokens(num_register_tokens=4))
    h = IMG_SIZE // PATCH_SIZE  # 4 (unchanged)
    assert out.shape == (B, FEATURES, h, h)


def test_reassemble_unsupported_scale_raises():
    with pytest.raises(ValueError, match="Unsupported layer_scale"):
        Reassemble(EMBED_DIM, FEATURES, PATCH_SIZE, IMG_SIZE, layer_scale=3)


# ---------------------------------------------------------------------------
# FeatureFusionBlock
# ---------------------------------------------------------------------------

def test_feature_fusion_block_single_input():
    """Single input: residual + 2× bilinear upsample."""
    block = FeatureFusionBlock(features=FEATURES)
    x = torch.zeros(B, FEATURES, 8, 8)
    out = block(x)
    assert out.shape == (B, FEATURES, 16, 16)


def test_feature_fusion_block_two_inputs():
    """Two inputs: skip connection before upsample."""
    block = FeatureFusionBlock(features=FEATURES)
    x0 = torch.zeros(B, FEATURES, 8, 8)
    x1 = torch.zeros(B, FEATURES, 8, 8)
    out = block(x0, x1)
    assert out.shape == (B, FEATURES, 16, 16)


# ---------------------------------------------------------------------------
# DPTDecoder (full pipeline)
# ---------------------------------------------------------------------------

def test_dpt_decoder_forward():
    """Full DPTDecoder forward pass produces the correct output shape."""
    decoder = DPTDecoder(
        embed_dim=EMBED_DIM,
        features=FEATURES,
        num_layers=4,
        patch_size=PATCH_SIZE,
        img_size=IMG_SIZE,
        out_channels=1,
    )

    hidden_states = [_make_tokens() for _ in range(4)]
    out = decoder(hidden_states)

    # Head: Conv → Upsample(2×) → Conv, starting from 32×32 → 64×64
    assert out.shape[0] == B
    assert out.shape[1] == 1
    # Spatial output = IMG_SIZE
    assert out.shape[2] == IMG_SIZE
    assert out.shape[3] == IMG_SIZE


def test_dpt_decoder_wrong_num_states_raises():
    """Mismatched number of hidden states raises AssertionError."""
    decoder = DPTDecoder(embed_dim=EMBED_DIM, features=FEATURES, num_layers=4,
                         patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    with pytest.raises(AssertionError):
        decoder([_make_tokens()] * 3)   # expected 4, got 3

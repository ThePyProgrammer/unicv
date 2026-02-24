"""Tests for SDTHead (unicv.nn.sdt)."""

import pytest
import torch

from unicv.nn.sdt import SDTHead


# ---------------------------------------------------------------------------
# Shared constants (small values for speed)
# ---------------------------------------------------------------------------

EMBED_DIM = 128
DECODER_DIM = 64
PATCH_SIZE = 14
IMG_SIZE = 56     # → h_base = 56 // 14 = 4  (4×4 patch grid, N=16 tokens)
NUM_LEVELS = 4
NUM_HEADS = 8     # 64 / 8 = 8 per head – valid
B = 2


def _make_hidden_states(num_register_tokens: int = 0) -> list[torch.Tensor]:
    """Return fake ViT hidden-state list (one per level)."""
    N = (IMG_SIZE // PATCH_SIZE) ** 2  # 16
    seq_len = 1 + num_register_tokens + N
    return [torch.zeros(B, seq_len, EMBED_DIM) for _ in range(NUM_LEVELS)]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_sdt_head_instantiation():
    head = SDTHead(
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        num_levels=NUM_LEVELS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        img_size=IMG_SIZE,
    )
    assert head.num_levels == NUM_LEVELS
    assert len(head.level_projectors) == NUM_LEVELS
    assert len(head.fuse_blocks) == NUM_LEVELS


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_sdt_head_forward_shape():
    """Output should be (B, 1, IMG_SIZE, IMG_SIZE)."""
    head = SDTHead(
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        num_levels=NUM_LEVELS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        img_size=IMG_SIZE,
        out_channels=1,
    )

    out = head(_make_hidden_states())
    assert out.shape == (B, 1, IMG_SIZE, IMG_SIZE)


def test_sdt_head_forward_non_negative():
    """ReLU at the end ensures all predictions are ≥ 0."""
    head = SDTHead(
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        num_levels=NUM_LEVELS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        img_size=IMG_SIZE,
    )
    out = head(_make_hidden_states())
    assert (out >= 0).all()


def test_sdt_head_wrong_num_levels_raises():
    """Passing the wrong number of hidden states raises AssertionError."""
    head = SDTHead(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
                   num_levels=4, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    with pytest.raises(AssertionError):
        head([torch.zeros(B, 17, EMBED_DIM)] * 3)  # expected 4, got 3

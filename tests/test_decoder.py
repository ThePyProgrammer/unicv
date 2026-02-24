"""Tests for MultiresConvDecoder and FeatureFusionBlock2d (unicv.nn.decoder)."""

import pytest
import torch

from unicv.nn.decoder import FeatureFusionBlock2d, MultiresConvDecoder


# ---------------------------------------------------------------------------
# FeatureFusionBlock2d
# ---------------------------------------------------------------------------

def test_feature_fusion_block2d_no_skip():
    """Single-input path: no skip connection, no deconv."""
    block = FeatureFusionBlock2d(num_features=64)
    x = torch.zeros(2, 64, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 16, 16)


def test_feature_fusion_block2d_with_skip():
    """Two-input path: skip connection fused before second residual block."""
    block = FeatureFusionBlock2d(num_features=64)
    x0 = torch.zeros(2, 64, 16, 16)
    x1 = torch.zeros(2, 64, 16, 16)
    out = block(x0, x1)
    assert out.shape == (2, 64, 16, 16)


def test_feature_fusion_block2d_with_deconv():
    """Deconv flag doubles the spatial size."""
    block = FeatureFusionBlock2d(num_features=64, deconv=True)
    x = torch.zeros(2, 64, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 32, 32)


def test_feature_fusion_block2d_with_batch_norm():
    """BatchNorm variant instantiates and runs without error."""
    block = FeatureFusionBlock2d(num_features=32, batch_norm=True)
    x = torch.zeros(2, 32, 8, 8)
    out = block(x)
    assert out.shape == (2, 32, 8, 8)


# ---------------------------------------------------------------------------
# MultiresConvDecoder
# ---------------------------------------------------------------------------

def test_multires_conv_decoder_shapes():
    """Verify output shapes for a 4-level multi-resolution decoder."""
    dims_encoder = [256, 512, 512, 512]   # finest → coarsest
    dim_decoder = 256

    decoder = MultiresConvDecoder(dims_encoder=dims_encoder, dim_decoder=dim_decoder)

    encodings = [
        torch.zeros(2, 256, 96, 96),  # finest
        torch.zeros(2, 512, 48, 48),
        torch.zeros(2, 512, 24, 24),
        torch.zeros(2, 512, 12, 12),  # coarsest
    ]

    features, lowres = decoder(encodings)

    assert features.shape == (2, dim_decoder, 96, 96)
    assert lowres.shape == (2, dim_decoder, 12, 12)


def test_multires_conv_decoder_wrong_number_of_encodings():
    """Passing the wrong number of encoder levels raises ValueError."""
    decoder = MultiresConvDecoder(dims_encoder=[64, 64, 64], dim_decoder=64)
    encodings = [torch.zeros(1, 64, 4, 4)]  # only 1 level, expected 3

    with pytest.raises(ValueError, match="expected levels"):
        decoder(encodings)

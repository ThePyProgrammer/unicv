"""Tests for FOVNetwork (unicv.nn.fov)."""

import torch

from unicv.nn.fov import FOVNetwork


# ---------------------------------------------------------------------------
# FOVNetwork (no extra ViT encoder)
# ---------------------------------------------------------------------------

def test_fov_network_no_encoder_instantiation():
    """FOVNetwork with no extra encoder creates correctly."""
    net = FOVNetwork(num_features=256)
    assert isinstance(net, FOVNetwork)
    assert not hasattr(net, "encoder")


def test_fov_network_no_encoder_forward():
    """With no encoder the image is ignored; lowres feature drives FOV.

    For num_features=256 the spatial pipeline is:
      Conv(stride=2) → Conv(stride=2) → Conv(stride=2) → Conv(kernel=6)
    Starting from h=48: 48 → 24 → 12 → 6 → (6-6+1)=1
    So the output should be (B, 1, 1, 1).
    """
    net = FOVNetwork(num_features=256)
    x = torch.zeros(2, 3, 64, 64)          # image – not used without encoder
    lowres = torch.zeros(2, 256, 48, 48)   # must be 48×48 for the conv stack

    out = net(x, lowres)
    assert out.shape == (2, 1, 1, 1)


def test_fov_network_no_encoder_batch_size_1():
    """FOVNetwork works for batch size 1."""
    net = FOVNetwork(num_features=128)
    x = torch.zeros(1, 3, 32, 32)
    lowres = torch.zeros(1, 128, 48, 48)
    out = net(x, lowres)
    assert out.shape[0] == 1
    assert out.shape[1] == 1

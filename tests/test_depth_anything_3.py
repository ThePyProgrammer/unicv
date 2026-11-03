"""Tests for DepthAnything3 and DINOv2Backbone (unicv.models.depth_anything_3)."""

from unittest.mock import patch

import torch
import torch.nn as nn

from unicv.models.depth_anything_3.model import (
    DepthAnything3,
    DepthAnything3Model,
    DINOv2Backbone,
)
from unicv.nn.dpt import DPTDecoder
from unicv.nn.gsdpt import GSDPT
from unicv.utils.types import Modality


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_dinov2(num_blocks: int = 24) -> nn.Module:
    """A minimal stand-in for a torch.hub DINOv2 model."""

    class _FakeDINOv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1)

    return _FakeDINOv2()


class _MockBackbone(nn.Module):
    """Lightweight backbone that mimics the DINOv2Backbone interface."""

    embed_dim: int = 256
    hook_layer_ids: list[int] = [0, 1, 2, 3]
    num_register_tokens: int = 0

    def __init__(self, img_size: int = 64, patch_size: int = 16):
        super().__init__()
        self._img_size = img_size
        self._patch_size = patch_size

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch_shape = x.shape[:2] if x.ndim == 5 else x.shape[:1]
        N = (self._img_size // self._patch_size) ** 2  # patch tokens
        # Each hidden state: [CLS, ...N patches...]
        return [torch.zeros(*batch_shape, 1 + N, self.embed_dim) for _ in range(4)]


# ---------------------------------------------------------------------------
# DINOv2Backbone
# ---------------------------------------------------------------------------

def test_dinov2_backbone_instantiation():
    """DINOv2Backbone can be constructed when torch.hub.load is mocked."""
    fake_dinov2 = _make_mock_dinov2(num_blocks=24)

    with patch("torch.hub.load", return_value=fake_dinov2):
        backbone = DINOv2Backbone(variant="vit_l", pretrained=False)

    assert backbone.embed_dim == 1024
    # Default hook_layer_ids for 24 blocks: [5, 11, 17, 23]
    assert len(backbone.hook_layer_ids) == 4


def test_dinov2_backbone_small_variant():
    """vit_s variant produces the correct embed_dim."""
    fake_dinov2 = _make_mock_dinov2(num_blocks=12)

    with patch("torch.hub.load", return_value=fake_dinov2):
        backbone = DINOv2Backbone(variant="vit_s", pretrained=False)

    assert backbone.embed_dim == 384


def test_dinov2_backbone_camera_token_and_cross_view_attention():
    class _MeanBlock(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + x.mean(dim=1, keepdim=True)

    class _CrossViewDINO(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_MeanBlock(), _MeanBlock()])

        def prepare_tokens_with_masks(self, x: torch.Tensor) -> torch.Tensor:
            value = x.mean(dim=(1, 2, 3), keepdim=False)[:, None, None]
            return value.expand(-1, 2, 1024).clone()

    fake_dinov2 = _CrossViewDINO()
    with patch("torch.hub.load", return_value=fake_dinov2):
        backbone = DINOv2Backbone(
            variant="vit_l",
            pretrained=False,
            hook_layer_ids=[1],
            alt_start=1,
        )

    assert backbone.model.camera_token.shape == (1, 2, 1024)
    images = torch.stack(
        (torch.zeros(3, 8, 8), torch.ones(3, 8, 8)), dim=0
    ).unsqueeze(0)
    together = backbone(images)[0]
    changed_second_view = images.clone()
    changed_second_view[:, 1] = 3
    changed = backbone(changed_second_view)[0]

    assert together.shape == (1, 2, 2, 2048)
    assert not torch.allclose(together[:, 0], changed[:, 0])


# ---------------------------------------------------------------------------
# DepthAnything3 – manual construction
# ---------------------------------------------------------------------------

def test_depth_anything3_instantiation():
    """DepthAnything3 can be built from a mock backbone and a DPTDecoder."""
    backbone = _MockBackbone(img_size=64, patch_size=16)
    decoder = DPTDecoder(
        embed_dim=256,
        features=64,
        num_layers=4,
        patch_size=16,
        img_size=64,
        out_channels=1,
    )
    model = DepthAnything3(backbone=backbone, decoder=decoder)
    assert isinstance(model, DepthAnything3)


def test_depth_anything3_forward():
    """DepthAnything3.forward produces a (B, 1, H, W) depth map."""
    backbone = _MockBackbone(img_size=64, patch_size=16)
    decoder = DPTDecoder(
        embed_dim=256,
        features=64,
        num_layers=4,
        patch_size=16,
        img_size=64,
        out_channels=1,
    )
    model = DepthAnything3(backbone=backbone, decoder=decoder)
    model.eval()

    x = torch.zeros(2, 3, 64, 64)
    out = model(x)

    assert out.shape == (2, 1, 64, 64)


# ---------------------------------------------------------------------------
# DepthAnything3 – from_config (requires mocked torch.hub)
# ---------------------------------------------------------------------------

def test_depth_anything3_from_config():
    """from_config builds the full model via the mocked DINOv2 hub."""
    # Use vit_s (12 blocks, embed_dim=384) to minimise parameter allocation.
    fake_dinov2 = _make_mock_dinov2(num_blocks=12)

    with patch("torch.hub.load", return_value=fake_dinov2):
        model = DepthAnything3.from_config(
            variant="vit_s",
            img_size=64,
            features=64,
            patch_size=14,
            pretrained=False,
        )

    assert isinstance(model, DepthAnything3)
    assert isinstance(model.backbone, DINOv2Backbone)
    assert isinstance(model.decoder, DPTDecoder)
    assert model.backbone.alt_start == 6
    assert model.decoder.reassemble_blocks[0].project.in_channels == 768


def test_vit_g_from_config_includes_gaussian_head():
    fake_dinov2 = _make_mock_dinov2(num_blocks=12)
    with patch("torch.hub.load", return_value=fake_dinov2):
        model = DepthAnything3.from_config(
            variant="vit_g",
            img_size=64,
            features=32,
            pretrained=False,
        )
    assert isinstance(model.gaussian_decoder, GSDPT)


# ---------------------------------------------------------------------------
# DepthAnything3Model – VisionModule wrapper
# ---------------------------------------------------------------------------

def test_depth_anything3_model_instantiation():
    """DepthAnything3Model wraps DepthAnything3 without error."""
    backbone = _MockBackbone(img_size=64, patch_size=16)
    decoder = DPTDecoder(
        embed_dim=256,
        features=64,
        num_layers=4,
        patch_size=16,
        img_size=64,
        out_channels=1,
    )
    net = DepthAnything3(backbone=backbone, decoder=decoder)
    wrapper = DepthAnything3Model(net=net)
    assert isinstance(wrapper, DepthAnything3Model)


def test_depth_anything3_model_forward():
    """DepthAnything3Model.__call__ runs full VisionModule pipeline."""
    backbone = _MockBackbone(img_size=64, patch_size=16)
    decoder = DPTDecoder(
        embed_dim=256,
        features=64,
        num_layers=4,
        patch_size=16,
        img_size=64,
        out_channels=1,
    )
    net = DepthAnything3(backbone=backbone, decoder=decoder)
    wrapper = DepthAnything3Model(net=net)
    net.eval()
    result = wrapper(rgb=torch.zeros(2, 3, 64, 64))
    assert Modality.DEPTH in result
    assert result[Modality.DEPTH].shape[0] == 2


def test_depth_anything3_multiview_and_gaussian_forward():
    backbone = _MockBackbone(img_size=64, patch_size=16)
    decoder = DPTDecoder(
        embed_dim=256,
        features=32,
        num_layers=4,
        patch_size=16,
        img_size=64,
        out_channels=1,
    )
    gaussian_decoder = GSDPT(
        embed_dim=256,
        features=32,
        num_layers=4,
        patch_size=16,
        img_size=64,
        sh_degree=0,
    )
    model = DepthAnything3(backbone, decoder, gaussian_decoder).eval()
    images = torch.zeros(1, 2, 3, 64, 64)
    prediction = model(images, predict_gaussians=True)

    assert prediction[Modality.DEPTH].shape == (1, 2, 1, 64, 64)
    assert prediction[Modality.SPLAT].xyz.shape == (1, 2, 64 * 64, 3)

"""Tests for VisionModule.from_pretrained classmethods.

Mocking strategy
----------------
``huggingface_hub`` and ``timm`` are installed; we patch individual functions
rather than injecting fake module objects:

* ``huggingface_hub.hf_hub_download``  – patched to return a sentinel path,
  preventing any real network download.
* ``timm.create_model``                – patched with a factory that returns
  tiny ViT stubs, preventing heavy model construction / weight download.
* ``torch.hub.load``                   – patched to return a tiny DINOv2 stub,
  preventing the real ``facebookresearch/dinov2`` download.
* ``torch.load``                       – patched to return a small dict,
  preventing real checkpoint I/O.

For DA3 the dummy checkpoint path ends in ``.pt`` (not ``.safetensors``) so
the code falls through to ``torch.load`` and safetensors is never exercised.

"Missing dependency" tests use ``patch.dict("sys.modules", {"pkg": None})``
which is the standard CPython mechanism for simulating an absent package.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared sentinel paths
# ---------------------------------------------------------------------------

_DUMMY_PT_PATH = "/tmp/fake_model.pt"


# ---------------------------------------------------------------------------
# Tiny model stubs
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    """Minimal patch-embed stub compatible with DepthProEncoder."""

    def __init__(self, img_size: int = 384, patch_size: int = 16) -> None:
        super().__init__()
        self.img_size   = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)


class _MockTimmViT(nn.Module):
    """Minimal timm ViT stub that satisfies DepthProEncoder attribute access."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_embed = _PatchEmbed()
        self.blocks      = nn.ModuleList([nn.Identity() for _ in range(24)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B   = x.shape[0]
        n   = (self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]) ** 2
        return torch.zeros(B, 1 + n, self.embed_dim)


class _MockDINOv2(nn.Module):
    """Minimal DINOv2 stub – DINOv2Backbone registers hooks on ``.blocks``."""

    def __init__(self, embed_dim: int = 64, num_blocks: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks    = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.embed_dim)


# ---------------------------------------------------------------------------
# timm side-effect factory
# ---------------------------------------------------------------------------

def _timm_factory(model_name: str, **kw) -> _MockTimmViT:
    """Return a mock ViT sized by the model name."""
    if "large" in model_name:
        return _MockTimmViT(embed_dim=64)
    return _MockTimmViT(embed_dim=32)


# ---------------------------------------------------------------------------
# DepthProModel.from_pretrained
# ---------------------------------------------------------------------------

class TestDepthProFromPretrained:

    def _run(self, sd: dict | None = None, use_fov_head: bool = True):
        from unicv.models.depth_pro.model import DepthProModel
        sd = sd or {}
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH), \
             patch("timm.create_model", side_effect=_timm_factory), \
             patch("torch.load", return_value=sd):
            return DepthProModel.from_pretrained(use_fov_head=use_fov_head)

    # -- type --

    def test_returns_depth_pro_model(self):
        from unicv.models.depth_pro.model import DepthProModel
        assert isinstance(self._run(), DepthProModel)

    # -- HuggingFace Hub calls --

    def test_calls_correct_repo_id(self):
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH) as mock_dl, \
             patch("timm.create_model", side_effect=_timm_factory), \
             patch("torch.load", return_value={}):
            from unicv.models.depth_pro.model import DepthProModel
            DepthProModel.from_pretrained()
        assert mock_dl.call_args.kwargs["repo_id"] == "apple/DepthPro"

    def test_calls_correct_filename(self):
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH) as mock_dl, \
             patch("timm.create_model", side_effect=_timm_factory), \
             patch("torch.load", return_value={}):
            from unicv.models.depth_pro.model import DepthProModel
            DepthProModel.from_pretrained()
        assert mock_dl.call_args.kwargs["filename"] == "depth_pro.pt"

    # -- optional dependency errors --

    def test_missing_huggingface_hub_raises(self):
        from unicv.models.depth_pro.model import DepthProModel
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                DepthProModel.from_pretrained()

    def test_missing_timm_raises(self):
        from unicv.models.depth_pro.model import DepthProModel
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH), \
             patch.dict("sys.modules", {"timm": None}), \
             patch("torch.load", return_value={}):
            with pytest.raises(ImportError, match="timm"):
                DepthProModel.from_pretrained()

    # -- architecture options --

    def test_no_fov_head(self):
        assert not hasattr(self._run(use_fov_head=False).net, "fov")

    def test_with_fov_head(self):
        assert hasattr(self._run(use_fov_head=True).net, "fov")

    def test_decoder_is_present(self):
        assert self._run().net.decoder is not None

    def test_encoder_is_present(self):
        assert self._run().net.encoder is not None


# ---------------------------------------------------------------------------
# DepthAnything3Model.from_pretrained
#
# Dummy path ends in .pt so from_pretrained uses torch.load (not safetensors).
# ---------------------------------------------------------------------------

def _da3_sd(variant: str = "vit_l") -> dict[str, torch.Tensor]:
    """Minimal DA3-format state dict for key-remapping tests.

    All conv weights use ``features=256`` output channels to match the unicv
    ``DPTDecoder`` where ``Reassemble.project`` is ``Conv2d(embed_dim, 256, 1)``.
    """
    D = {"vit_s": 384, "vit_b": 768, "vit_l": 1024, "vit_g": 1536}[variant]
    F = 256
    return {
        # backbone
        "pretrained.norm.weight": torch.ones(D),
        "pretrained.norm.bias":   torch.zeros(D),
        # reassemble project (index 0) — out_channels must match features=256
        "depth_head.projects.0.weight": torch.zeros(F, D, 1, 1),
        "depth_head.projects.0.bias":   torch.zeros(F),
        # fusion block (refinenet4 → fusion_blocks[0])
        "depth_head.scratch.refinenet4.resConfUnit1.conv1.weight": torch.zeros(F, F, 3, 3),
        "depth_head.scratch.refinenet4.resConfUnit1.conv1.bias":   torch.zeros(F),
        "depth_head.scratch.refinenet4.out_conv.weight": torch.zeros(F, F, 1, 1),
        "depth_head.scratch.refinenet4.out_conv.bias":   torch.ones(F),
        # head conv1
        "depth_head.scratch.output_conv1.0.weight": torch.zeros(F // 2, F, 3, 3),
        "depth_head.scratch.output_conv1.0.bias":   torch.ones(F // 2),
    }


class TestDepthAnything3FromPretrained:

    def _run(self, variant: str = "vit_l", sd: dict | None = None):
        from unicv.models.depth_anything_3.model import DepthAnything3Model
        embed     = {"vit_s": 384, "vit_b": 768, "vit_l": 1024, "vit_g": 1536}[variant]
        mock_dino = _MockDINOv2(embed_dim=embed, num_blocks=12)
        sd        = sd if sd is not None else _da3_sd(variant)
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH), \
             patch("torch.hub.load", return_value=mock_dino), \
             patch("torch.load", return_value=sd):
            return DepthAnything3Model.from_pretrained(variant=variant)

    # -- type --

    def test_returns_correct_type(self):
        from unicv.models.depth_anything_3.model import DepthAnything3Model
        assert isinstance(self._run(), DepthAnything3Model)

    # -- architecture per variant --

    def test_vit_l_embed_dim(self):
        assert self._run("vit_l").net.backbone.embed_dim == 1024

    def test_vit_s_embed_dim(self):
        assert self._run("vit_s").net.backbone.embed_dim == 384

    def test_vit_b_embed_dim(self):
        assert self._run("vit_b").net.backbone.embed_dim == 768

    # -- HuggingFace Hub repo IDs --

    def _get_repo_id(self, variant: str) -> str:
        embed     = {"vit_s": 384, "vit_b": 768, "vit_l": 1024, "vit_g": 1536}[variant]
        mock_dino = _MockDINOv2(embed_dim=embed)
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH) as mock_dl, \
             patch("torch.hub.load", return_value=mock_dino), \
             patch("torch.load", return_value={}):
            from unicv.models.depth_anything_3.model import DepthAnything3Model
            DepthAnything3Model.from_pretrained(variant=variant)
        return mock_dl.call_args.kwargs["repo_id"]

    def test_repo_id_vit_l(self):
        assert self._get_repo_id("vit_l") == "depth-anything/DA3-LARGE"

    def test_repo_id_vit_s(self):
        assert self._get_repo_id("vit_s") == "depth-anything/DA3-SMALL"

    def test_repo_id_vit_b(self):
        assert self._get_repo_id("vit_b") == "depth-anything/DA3-BASE"

    def test_repo_id_vit_g(self):
        assert self._get_repo_id("vit_g") == "depth-anything/DA3-GIANT"

    # -- guard clauses --

    def test_invalid_variant_raises(self):
        from unicv.models.depth_anything_3.model import DepthAnything3Model
        with pytest.raises(ValueError, match="variant must be one of"):
            DepthAnything3Model.from_pretrained("vit_xxl")

    def test_missing_huggingface_hub_raises(self):
        from unicv.models.depth_anything_3.model import DepthAnything3Model
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                DepthAnything3Model.from_pretrained()

    # -- key remapping --

    def test_backbone_key_remapped(self):
        """pretrained.norm.weight → backbone.model.norm.weight (no crash)."""
        sd    = {"pretrained.norm.weight": torch.full((1024,), 7.0),
                 "pretrained.norm.bias":   torch.zeros(1024)}
        model = self._run("vit_l", sd=sd)
        assert model.net.backbone is not None

    def test_fusion_block_remapping(self):
        """refinenet4.out_conv → fusion_blocks[0].out_conv bias filled with 1s."""
        sd    = {"depth_head.scratch.refinenet4.out_conv.weight": torch.zeros(256, 256, 1, 1),
                 "depth_head.scratch.refinenet4.out_conv.bias":   torch.ones(256)}
        model = self._run("vit_l", sd=sd)
        assert torch.all(model.net.decoder.fusion_blocks[0].out_conv.bias == 1.0)

    def test_head_output_conv1_remapping(self):
        """depth_head.scratch.output_conv1.bias → decoder.head[0].bias."""
        sd    = {"depth_head.scratch.output_conv1.weight": torch.zeros(128, 256, 3, 3),
                 "depth_head.scratch.output_conv1.bias":   torch.full((128,), 3.0)}
        model = self._run("vit_l", sd=sd)
        assert torch.all(model.net.decoder.head[0].bias == 3.0)

    def test_reassemble_project_remapping(self):
        """depth_head.projects.0.bias → decoder.reassemble_blocks[0].project.bias."""
        sd    = {"depth_head.projects.0.weight": torch.zeros(256, 1024, 1, 1),
                 "depth_head.projects.0.bias":   torch.full((256,), 5.0)}
        model = self._run("vit_l", sd=sd)
        assert torch.all(model.net.decoder.reassemble_blocks[0].project.bias == 5.0)

    def test_resconfunit_renamed_to_resconvunit(self):
        """resConfUnit keys are remapped to resConvUnit without error."""
        sd = {
            "depth_head.scratch.refinenet3.resConfUnit2.conv1.weight": torch.zeros(256, 256, 3, 3),
            "depth_head.scratch.refinenet3.resConfUnit2.conv1.bias":   torch.zeros(256),
        }
        model = self._run("vit_l", sd=sd)
        assert isinstance(model.net.decoder.fusion_blocks[1].resConvUnit2.conv1.weight, torch.Tensor)


# ---------------------------------------------------------------------------
# CameraDepthModel.from_pretrained
# ---------------------------------------------------------------------------

def _cdm_sd() -> dict[str, torch.Tensor]:
    """Minimal CDM-format state dict for key-remapping tests."""
    D = 1024   # vit_l embed_dim
    F = 256
    return {
        # RGB backbone stored as 'pretrained.*'
        "pretrained.norm.weight": torch.ones(D),
        "pretrained.norm.bias":   torch.zeros(D),
        # Depth backbone stored as 'depth_encoder.*'
        "depth_encoder.norm.weight": torch.ones(D),
        "depth_encoder.norm.bias":   torch.zeros(D),
        # Fusion block (refinenet4 → fusion_blocks[0])
        "depth_head.scratch.refinenet4.out_conv.weight": torch.zeros(F, F, 1, 1),
        "depth_head.scratch.refinenet4.out_conv.bias":   torch.ones(F),
    }


class TestCameraDepthModelFromPretrained:

    def _run(self, camera: str = "d405", sd: dict | None = None):
        from unicv.models.cdm.model import CameraDepthModel
        mock_dino = _MockDINOv2(embed_dim=1024, num_blocks=12)
        sd        = sd if sd is not None else _cdm_sd()
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH), \
             patch("torch.hub.load", return_value=mock_dino), \
             patch("torch.load", return_value=sd):
            return CameraDepthModel.from_pretrained(camera=camera)

    # -- type --

    def test_returns_correct_type(self):
        from unicv.models.cdm.model import CameraDepthModel
        assert isinstance(self._run(), CameraDepthModel)

    # -- HuggingFace Hub calls --

    def _get_hub_kwargs(self, camera: str = "d405") -> dict:
        mock_dino = _MockDINOv2(embed_dim=1024)
        with patch("huggingface_hub.hf_hub_download", return_value=_DUMMY_PT_PATH) as mock_dl, \
             patch("torch.hub.load", return_value=mock_dino), \
             patch("torch.load", return_value={}):
            from unicv.models.cdm.model import CameraDepthModel
            CameraDepthModel.from_pretrained(camera=camera)
        return mock_dl.call_args.kwargs

    def test_repo_id_d405(self):
        assert self._get_hub_kwargs("d405")["repo_id"] == "depth-anything/camera-depth-model-d405"

    def test_repo_id_d435(self):
        assert self._get_hub_kwargs("d435")["repo_id"] == "depth-anything/camera-depth-model-d435"

    def test_repo_id_l515(self):
        assert self._get_hub_kwargs("l515")["repo_id"] == "depth-anything/camera-depth-model-l515"

    def test_repo_id_kinect(self):
        assert self._get_hub_kwargs("kinect")["repo_id"] == "depth-anything/camera-depth-model-kinect"

    def test_filename_is_model_pth(self):
        assert self._get_hub_kwargs()["filename"] == "model.pth"

    # -- guard clauses --

    def test_invalid_camera_raises(self):
        from unicv.models.cdm.model import CameraDepthModel
        with pytest.raises(ValueError, match="camera must be one of"):
            CameraDepthModel.from_pretrained(camera="realsense")

    def test_missing_huggingface_hub_raises(self):
        from unicv.models.cdm.model import CameraDepthModel
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                CameraDepthModel.from_pretrained()

    # -- all camera variants --

    def test_d435_camera(self):
        from unicv.models.cdm.model import CameraDepthModel
        assert isinstance(self._run("d435"), CameraDepthModel)

    def test_l515_camera(self):
        from unicv.models.cdm.model import CameraDepthModel
        assert isinstance(self._run("l515"), CameraDepthModel)

    def test_kinect_camera(self):
        from unicv.models.cdm.model import CameraDepthModel
        assert isinstance(self._run("kinect"), CameraDepthModel)

    # -- key remapping --

    def test_rgb_backbone_pretrained_remapping(self):
        """'pretrained.*' → rgb_backbone.model.*  (no crash)."""
        sd    = {"pretrained.norm.weight": torch.ones(1024),
                 "pretrained.norm.bias":   torch.zeros(1024)}
        assert self._run(sd=sd).net.rgb_backbone is not None

    def test_depth_backbone_remapping(self):
        """'depth_encoder.*' → depth_backbone.model.*  (no crash)."""
        sd    = {"depth_encoder.norm.weight": torch.ones(1024),
                 "depth_encoder.norm.bias":   torch.zeros(1024)}
        assert self._run(sd=sd).net.depth_backbone is not None

    def test_fusion_block_remapping(self):
        """refinenet4 → decoder.fusion_blocks[0].out_conv bias filled with 2s."""
        sd    = {"depth_head.scratch.refinenet4.out_conv.weight": torch.zeros(256, 256, 1, 1),
                 "depth_head.scratch.refinenet4.out_conv.bias":   torch.full((256,), 2.0)}
        model = self._run(sd=sd)
        assert torch.all(model.net.decoder.fusion_blocks[0].out_conv.bias == 2.0)

    def test_head_output_conv1_remapping(self):
        """depth_head.scratch.output_conv1.bias → decoder.head[0].bias."""
        sd    = {"depth_head.scratch.output_conv1.weight": torch.zeros(128, 256, 3, 3),
                 "depth_head.scratch.output_conv1.bias":   torch.full((128,), 4.0)}
        model = self._run(sd=sd)
        assert torch.all(model.net.decoder.head[0].bias == 4.0)


# ---------------------------------------------------------------------------
# SHARPModel.from_pretrained
#
# The official checkpoint is fetched via torch.hub.load_state_dict_from_url
# (Apple CDN, not HuggingFace).  torch.hub.load is also mocked to prevent
# the DINOv2 backbone from being downloaded during SHARP.from_config.
# ---------------------------------------------------------------------------

_SHARP_CKPT_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
)

_F = 256   # default features width


def _sharp_sd() -> dict[str, torch.Tensor]:
    """Minimal official-format state dict for SHARP remapping tests.

    Uses shapes compatible with our DPTDecoder (features=256) so the
    shape-filter inside from_pretrained keeps them.
    """
    return {
        # Fusion block 0 – resnet1 conv1
        "gaussian_decoder.fusions.0.resnet1.residual.1.weight": torch.full((_F, _F, 3, 3), 7.0),
        "gaussian_decoder.fusions.0.resnet1.residual.1.bias":   torch.zeros(_F),
        # Fusion block 0 – resnet1 conv2
        "gaussian_decoder.fusions.0.resnet1.residual.3.weight": torch.full((_F, _F, 3, 3), 2.0),
        "gaussian_decoder.fusions.0.resnet1.residual.3.bias":   torch.zeros(_F),
        # Fusion block 0 – out_conv
        "gaussian_decoder.fusions.0.out_conv.weight": torch.full((_F, _F, 1, 1), 5.0),
        "gaussian_decoder.fusions.0.out_conv.bias":   torch.zeros(_F),
        # Fusion block 1 – resnet2 conv1
        "gaussian_decoder.fusions.1.resnet2.residual.1.weight": torch.full((_F, _F, 3, 3), 3.0),
        "gaussian_decoder.fusions.1.resnet2.residual.1.bias":   torch.zeros(_F),
        # Geometry prediction head – 6 output channels (3×num_layers=2)
        # Shape mismatch with our xyz_head (3 channels): should be filtered.
        "prediction_head.geometry_prediction_head.weight": torch.zeros(6, _F, 1, 1),
        "prediction_head.geometry_prediction_head.bias":   torch.zeros(6),
    }


class TestSHARPFromPretrained:

    def _run(self, sd: dict | None = None):
        from unicv.models.sharp.model import SHARPModel
        mock_dino = _MockDINOv2(embed_dim=1024, num_blocks=12)
        sd = sd if sd is not None else _sharp_sd()
        with patch("torch.hub.load_state_dict_from_url", return_value=sd), \
             patch("torch.hub.load", return_value=mock_dino):
            return SHARPModel.from_pretrained()

    # -- type --

    def test_returns_sharp_model(self):
        from unicv.models.sharp.model import SHARPModel
        assert isinstance(self._run(), SHARPModel)

    # -- download --

    def test_correct_url(self):
        mock_dino = _MockDINOv2(embed_dim=1024, num_blocks=12)
        with patch("torch.hub.load_state_dict_from_url", return_value={}) as mock_dl, \
             patch("torch.hub.load", return_value=mock_dino):
            from unicv.models.sharp.model import SHARPModel
            SHARPModel.from_pretrained()
        assert mock_dl.call_args.args[0] == _SHARP_CKPT_URL

    def test_cache_dir_forwarded(self):
        mock_dino = _MockDINOv2(embed_dim=1024, num_blocks=12)
        with patch("torch.hub.load_state_dict_from_url", return_value={}) as mock_dl, \
             patch("torch.hub.load", return_value=mock_dino):
            from unicv.models.sharp.model import SHARPModel
            SHARPModel.from_pretrained(cache_dir="/tmp/weights")
        assert mock_dl.call_args.kwargs["model_dir"] == "/tmp/weights"

    # -- key remapping: fusion blocks --

    def test_fusion_resnet1_conv1_remapped(self):
        """fusions.0.resnet1.residual.1.weight → fusion_blocks[0].resConvUnit1.conv1.weight"""
        sd    = {"gaussian_decoder.fusions.0.resnet1.residual.1.weight": torch.full((_F, _F, 3, 3), 7.0),
                 "gaussian_decoder.fusions.0.resnet1.residual.1.bias":   torch.zeros(_F)}
        model = self._run(sd)
        assert torch.all(model.net.feature_decoder.fusion_blocks[0].resConvUnit1.conv1.weight == 7.0)

    def test_fusion_resnet1_conv2_remapped(self):
        """fusions.0.resnet1.residual.3.weight → fusion_blocks[0].resConvUnit1.conv2.weight"""
        sd    = {"gaussian_decoder.fusions.0.resnet1.residual.3.weight": torch.full((_F, _F, 3, 3), 2.0),
                 "gaussian_decoder.fusions.0.resnet1.residual.3.bias":   torch.zeros(_F)}
        model = self._run(sd)
        assert torch.all(model.net.feature_decoder.fusion_blocks[0].resConvUnit1.conv2.weight == 2.0)

    def test_fusion_resnet2_conv1_remapped(self):
        """fusions.1.resnet2.residual.1.weight → fusion_blocks[1].resConvUnit2.conv1.weight"""
        sd    = {"gaussian_decoder.fusions.1.resnet2.residual.1.weight": torch.full((_F, _F, 3, 3), 3.0),
                 "gaussian_decoder.fusions.1.resnet2.residual.1.bias":   torch.zeros(_F)}
        model = self._run(sd)
        assert torch.all(model.net.feature_decoder.fusion_blocks[1].resConvUnit2.conv1.weight == 3.0)

    def test_fusion_out_conv_remapped(self):
        """fusions.0.out_conv.weight → fusion_blocks[0].out_conv.weight"""
        sd    = {"gaussian_decoder.fusions.0.out_conv.weight": torch.full((_F, _F, 1, 1), 5.0),
                 "gaussian_decoder.fusions.0.out_conv.bias":   torch.zeros(_F)}
        model = self._run(sd)
        assert torch.all(model.net.feature_decoder.fusion_blocks[0].out_conv.weight == 5.0)

    # -- shape filtering --

    def test_shape_mismatch_does_not_raise(self):
        """geometry_prediction_head with 6 output channels is silently dropped
        (our xyz_head has 3 output channels)."""
        sd = {
            "prediction_head.geometry_prediction_head.weight": torch.zeros(6, _F, 1, 1),
            "prediction_head.geometry_prediction_head.bias":   torch.zeros(6),
        }
        # Must not raise RuntimeError.
        self._run(sd)

    def test_geometry_head_loaded_when_shape_matches(self):
        """If the official head happens to have matching shape (3 ch), it loads."""
        sd    = {"prediction_head.geometry_prediction_head.weight": torch.full((3, _F, 1, 1), 9.0),
                 "prediction_head.geometry_prediction_head.bias":   torch.zeros(3)}
        model = self._run(sd)
        assert torch.all(model.net.gaussian_head.xyz_head.weight == 9.0)

    # -- warning on missing keys --

    def test_missing_keys_warn(self):
        """An empty state dict → all model keys are missing → UserWarning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._run({})
        assert any("SHARPModel" in str(x.message) for x in w)

"""Tests for DepthProEncoder, DepthPro, and DepthProModel (unicv.models.depth_pro)."""

import torch
import torch.nn as nn

from unicv.models.depth_pro.encoder import DepthProEncoder
from unicv.models.depth_pro.model import DepthPro, DepthProModel
from unicv.nn.decoder import MultiresConvDecoder


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockPatchEmbed(nn.Module):
    def __init__(self, img_size: int = 96, patch_size: int = 16):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)


class _MockViT(nn.Module):
    """Minimal ViT-like module whose forward returns (B, N+1, embed_dim) tokens."""

    def __init__(self, embed_dim: int = 128, img_size: int = 96,
                 patch_size: int = 16, num_blocks: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _MockPatchEmbed(img_size, patch_size)
        # Blocks must be an nn.ModuleList so hooks can be registered.
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self._out_size = img_size // patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        N = self._out_size * self._out_size
        return torch.zeros(B, N + 1, self.embed_dim)


def _make_encoder(embed_dim: int = 128) -> DepthProEncoder:
    """Return a DepthProEncoder built around two _MockViT backbones."""
    patch_enc = _MockViT(embed_dim=embed_dim, img_size=96, patch_size=16, num_blocks=8)
    image_enc = _MockViT(embed_dim=embed_dim, img_size=96, patch_size=16, num_blocks=8)
    return DepthProEncoder(
        dims_encoder=[embed_dim] * 4,
        patch_encoder=patch_enc,
        image_encoder=image_enc,
        hook_block_ids=[1, 3],
        decoder_features=embed_dim,
    )


# ---------------------------------------------------------------------------
# DepthProEncoder – instantiation and utility methods
# ---------------------------------------------------------------------------

def test_depth_pro_encoder_instantiation():
    encoder = _make_encoder()
    # out_size = img_size // patch_size = 96 // 16 = 6
    assert encoder.out_size == 6
    # img_size property = patch_embed.img_size[0] * 4
    assert encoder.img_size == 96 * 4


def test_depth_pro_encoder_split():
    """split() tiles a ≥384×384 image into 384×384 patches."""
    encoder = _make_encoder()
    x = torch.zeros(1, 3, 384, 384)
    patches = encoder.split(x, overlap_ratio=0.0)
    # A 384×384 image with patch_size=384 → exactly 1 patch
    assert patches.shape[0] == 1
    assert patches.shape[-2:] == (384, 384)


def test_depth_pro_encoder_reshape_feature():
    """reshape_feature drops the CLS token and reshapes to (B, D, h, w)."""
    encoder = _make_encoder()
    # out_size = 6, so 6*6 + 1 CLS = 37 tokens
    embeddings = torch.zeros(4, 37, 128)
    reshaped = encoder.reshape_feature(embeddings, 6, 6)
    assert reshaped.shape == (4, 128, 6, 6)


def test_depth_pro_encoder_merge():
    """merge() reconstructs a spatial map from a single 1×1 batch of patches."""
    encoder = _make_encoder()
    # Single-step merge (steps=1): output equals input.
    feature_patches = torch.zeros(1, 128, 6, 6)
    merged = encoder.merge(feature_patches, batch_size=1)
    assert merged.shape == (1, 128, 6, 6)


# ---------------------------------------------------------------------------
# DepthPro – instantiation and forward
# ---------------------------------------------------------------------------

class _TinyMockEncoder(nn.Module):
    """Returns three fixed-size feature maps; encoder.img_size = 32."""

    @property
    def img_size(self) -> int:
        return 32

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.shape[0]
        return [
            torch.zeros(B, 64, 8, 8),   # finest
            torch.zeros(B, 64, 4, 4),
            torch.zeros(B, 64, 2, 2),   # coarsest
        ]


def test_depth_pro_instantiation():
    encoder = _TinyMockEncoder()
    decoder = MultiresConvDecoder(dims_encoder=[64, 64, 64], dim_decoder=64)
    model = DepthPro(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
        use_fov_head=False,
    )
    assert isinstance(model, DepthPro)
    assert model.img_size == 32


def test_depth_pro_forward():
    """DepthPro.forward returns (canonical_inverse_depth, None) when no FOV head."""
    encoder = _TinyMockEncoder()
    decoder = MultiresConvDecoder(dims_encoder=[64, 64, 64], dim_decoder=64)
    model = DepthPro(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
        use_fov_head=False,
    )
    model.eval()

    x = torch.zeros(2, 3, 32, 32)  # matches img_size=32
    depth, fov = model(x)

    assert depth.shape[0] == 2
    assert depth.shape[1] == 1
    assert fov is None


# ---------------------------------------------------------------------------
# DepthProModel – VisionModule wrapper
# ---------------------------------------------------------------------------

def test_depth_pro_model_instantiation():
    encoder = _TinyMockEncoder()
    decoder = MultiresConvDecoder(dims_encoder=[64, 64, 64], dim_decoder=64)
    net = DepthPro(encoder=encoder, decoder=decoder, last_dims=(32, 1), use_fov_head=False)

    wrapper = DepthProModel(net=net)
    assert isinstance(wrapper, DepthProModel)

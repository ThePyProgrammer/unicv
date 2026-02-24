"""Tests for GaussianCloud and TriangleMesh (unicv.utils.structs)."""

import torch

from unicv.utils.structs import GaussianCloud, TriangleMesh


# ---------------------------------------------------------------------------
# GaussianCloud
# ---------------------------------------------------------------------------

def _make_cloud(N: int = 10, sh_degree: int = 0, batch: bool = False) -> GaussianCloud:
    K = (sh_degree + 1) ** 2
    shape = (2, N) if batch else (N,)
    def t(*extra): return torch.zeros(*shape, *extra)
    return GaussianCloud(
        xyz=t(3),
        scales=t(3),
        rotations=t(4),
        opacities=t(1),
        sh_coeffs=t(K, 3),
    )


def test_gaussian_cloud_num_gaussians():
    cloud = _make_cloud(N=16)
    assert cloud.num_gaussians == 16


def test_gaussian_cloud_sh_degree_0():
    cloud = _make_cloud(sh_degree=0)
    assert cloud.sh_degree == 0
    assert cloud.sh_coeffs.shape[-2] == 1


def test_gaussian_cloud_sh_degree_3():
    cloud = _make_cloud(sh_degree=3)
    assert cloud.sh_degree == 3
    assert cloud.sh_coeffs.shape[-2] == 16


def test_gaussian_cloud_batched_shapes():
    cloud = _make_cloud(N=8, sh_degree=1, batch=True)
    assert cloud.xyz.shape == (2, 8, 3)
    assert cloud.scales.shape == (2, 8, 3)
    assert cloud.rotations.shape == (2, 8, 4)
    assert cloud.opacities.shape == (2, 8, 1)
    assert cloud.sh_coeffs.shape == (2, 8, 4, 3)


def test_gaussian_cloud_to_device():
    cloud = _make_cloud(N=4)
    moved = cloud.to("cpu")          # stay on CPU; tests device round-trip
    assert moved.xyz.device.type == "cpu"


def test_gaussian_cloud_detach():
    cloud = _make_cloud(N=4)
    cloud.xyz.requires_grad_(True)
    det = cloud.detach()
    assert not det.xyz.requires_grad


# ---------------------------------------------------------------------------
# TriangleMesh
# ---------------------------------------------------------------------------

def _make_mesh(V: int = 8, F: int = 12, with_attrs: bool = False) -> TriangleMesh:
    return TriangleMesh(
        vertices=torch.zeros(V, 3),
        faces=torch.zeros(F, 3, dtype=torch.long),
        vertex_normals=torch.zeros(V, 3) if with_attrs else None,
        vertex_colors=torch.ones(V, 3) if with_attrs else None,
    )


def test_triangle_mesh_counts():
    mesh = _make_mesh(V=8, F=12)
    assert mesh.num_vertices == 8
    assert mesh.num_faces == 12


def test_triangle_mesh_optional_attrs_none():
    mesh = _make_mesh()
    assert mesh.vertex_normals is None
    assert mesh.vertex_colors is None


def test_triangle_mesh_optional_attrs_present():
    mesh = _make_mesh(with_attrs=True)
    assert mesh.vertex_normals is not None
    assert mesh.vertex_colors is not None
    assert mesh.vertex_colors.shape == (8, 3)


def test_triangle_mesh_to_device():
    mesh = _make_mesh(V=4, F=6, with_attrs=True)
    moved = mesh.to("cpu")
    assert moved.vertices.device.type == "cpu"
    assert moved.vertex_normals is not None


def test_triangle_mesh_to_device_without_attrs():
    mesh = _make_mesh(V=4, F=6, with_attrs=False)
    moved = mesh.to("cpu")
    assert moved.vertex_normals is None


def test_triangle_mesh_detach():
    mesh = _make_mesh(with_attrs=True)
    mesh.vertices.requires_grad_(True)
    det = mesh.detach()
    assert not det.vertices.requires_grad
    assert det.vertex_normals is not None

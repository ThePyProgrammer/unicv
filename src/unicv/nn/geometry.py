"""Differentiable camera-geometry utilities.

Provides two fundamental operations needed by cost-volume and splat models:

* :func:`backproject_depth` — lifts a depth map to a 3-D point cloud in camera
  coordinates, given a pinhole camera intrinsics matrix.
* :func:`homography_warp` — warps a source feature map into the reference view
  at a specified fronto-parallel depth plane, using the homography induced by
  the relative camera pose.  This is the core primitive for plane-sweep stereo.

All operations are fully differentiable with respect to their tensor inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Lift a depth map to 3-D camera-space coordinates.

    For a pinhole camera with focal lengths ``(fx, fy)`` and principal point
    ``(cx, cy)``, the un-projection formula for a pixel ``(u, v)`` at depth
    ``d`` is:

    .. code-block::

        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d

    Args:
        depth:      Depth map, shape ``(B, 1, H, W)`` or ``(B, H, W)``.
                    Values are metric distances along the optical axis (z).
        intrinsics: Pinhole intrinsics matrix, shape ``(B, 3, 3)`` or ``(3, 3)``.
                    Assumed layout::

                        [[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0,  1]]

    Returns:
        Point cloud in camera coordinates, shape ``(B, H, W, 3)``.
        The last dimension is ordered ``[x, y, z]``.
    """
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)            # (B, 1, H, W)

    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(B, -1, -1)

    intrinsics = intrinsics.to(dtype)

    fx = intrinsics[:, 0, 0].view(B, 1, 1)   # (B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    # Pixel coordinate grids: u = col index (x), v = row index (y).
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # each (H, W)

    d = depth[:, 0]           # (B, H, W)
    x = (u - cx) * d / fx     # (B, H, W)
    y = (v - cy) * d / fy     # (B, H, W)
    z = d                      # (B, H, W)

    return torch.stack([x, y, z], dim=-1)     # (B, H, W, 3)


def homography_warp(
    src_feats: torch.Tensor,
    depth_plane: float | torch.Tensor,
    ref_intrinsics: torch.Tensor,
    src_intrinsics: torch.Tensor,
    ref_to_src: torch.Tensor,
) -> torch.Tensor:
    """Warp source features into the reference view at a fronto-parallel depth.

    For a plane with normal ``n = [0, 0, 1]^T`` at depth ``d`` in the
    reference camera, the induced homography from reference to source is:

    .. code-block::

        H_d = K_src @ (R + t @ n^T / d) @ K_ref_inv

    where ``[R | t] = ref_to_src[:3, :]``.

    Args:
        src_feats:       Source feature map to warp.  Shape ``(B, C, H, W)``.
        depth_plane:     Scalar depth hypothesis ``d > 0`` (float or scalar
                         tensor).  Depth is measured along the reference
                         camera's optical axis.
        ref_intrinsics:  Reference camera intrinsics ``K_ref``.
                         Shape ``(B, 3, 3)`` or ``(3, 3)``.
        src_intrinsics:  Source camera intrinsics ``K_src``.
                         Shape ``(B, 3, 3)`` or ``(3, 3)``.
        ref_to_src:      4×4 rigid transform from reference to source camera
                         coordinates ``T_{src←ref}``.  Shape ``(B, 4, 4)``.

    Returns:
        Warped source feature map in the reference view.
        Shape ``(B, C, H, W)``.  Out-of-bounds regions are filled with zero.
    """
    B, C, H, W = src_feats.shape
    device, dtype = src_feats.device, src_feats.dtype

    # Broadcast intrinsics.
    if ref_intrinsics.dim() == 2:
        ref_intrinsics = ref_intrinsics.unsqueeze(0).expand(B, -1, -1)
    if src_intrinsics.dim() == 2:
        src_intrinsics = src_intrinsics.unsqueeze(0).expand(B, -1, -1)

    ref_K = ref_intrinsics.to(dtype)
    src_K = src_intrinsics.to(dtype)

    R = ref_to_src[:, :3, :3].to(dtype)     # (B, 3, 3)
    t = ref_to_src[:, :3, 3:4].to(dtype)    # (B, 3, 1)

    # Scalar depth → (B, 1, 1) for broadcasting.
    if isinstance(depth_plane, torch.Tensor):
        d = depth_plane.to(dtype=dtype, device=device).reshape(B, 1, 1)
    else:
        d = torch.tensor(depth_plane, dtype=dtype, device=device).view(1, 1, 1).expand(B, -1, -1)

    # n^T = [0, 0, 1]; t @ n^T = outer product (B, 3, 3).
    n_row = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)   # (1, 3)
    n_row = n_row.unsqueeze(0).expand(B, 1, 3)                            # (B, 1, 3)
    outer = torch.bmm(t, n_row)                                           # (B, 3, 3)

    # Unnormalised homography: R + t*n^T / d.
    H_unnorm = R + outer / d                                              # (B, 3, 3)

    # Full homography: K_src @ H_unnorm @ K_ref_inv.
    ref_K_inv = torch.linalg.inv(ref_K)                                   # (B, 3, 3)
    Hmat = src_K @ H_unnorm @ ref_K_inv                                   # (B, 3, 3)

    # Build a pixel coordinate grid for the reference view.
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # (H, W) each
    ones = torch.ones(H * W, 1, device=device, dtype=dtype)
    # Homogeneous reference pixel coords: [x=col, y=row, 1].
    coords_ref = torch.stack(
        [u_grid.flatten(), v_grid.flatten(), ones.squeeze(1)], dim=1
    )                                                                      # (H*W, 3)
    coords_ref = coords_ref.unsqueeze(0).expand(B, -1, -1)                # (B, H*W, 3)

    # Apply homography: (B, 3, 3) × (B, 3, N) → (B, 3, N) → (B, N, 3).
    coords_src = torch.bmm(
        Hmat, coords_ref.permute(0, 2, 1)
    ).permute(0, 2, 1)                                                     # (B, H*W, 3)

    # Perspective divide.
    w = coords_src[..., 2:3].clamp(min=1e-8)
    coords_src_px = coords_src[..., :2] / w                               # (B, H*W, 2)

    # Normalise to [-1, 1] for grid_sample (x = col, y = row).
    x_norm = (coords_src_px[..., 0] / (W - 1)) * 2.0 - 1.0
    y_norm = (coords_src_px[..., 1] / (H - 1)) * 2.0 - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(B, H, W, 2)     # (B, H, W, 2)

    return F.grid_sample(
        src_feats,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


__all__ = ["backproject_depth", "homography_warp"]

"""
Lightweight reimplementation of pytorch3d.transforms.
Only the functions used by GVHMR are included.
"""
import math

import torch
import torch.nn.functional as F


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=flr.item()))
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
    ].reshape(batch_dim + (4,))


def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half = torch.where(
        small_angles,
        0.5 * torch.ones_like(angles),
        torch.sin(half_angles) / angles.clamp(min=eps),
    )
    cos_half = torch.where(small_angles, torch.ones_like(angles), torch.cos(half_angles))
    return torch.cat([cos_half, axis_angle * sin_half], dim=-1)


def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    sin_half_angles = norms.clamp(min=eps)
    return quaternions[..., 1:] * (angles / sin_half_angles)


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


# ── so3_exp_map / so3_log_map (Rodrigues formula) ──────────────────────


def _hat(v):
    """Skew-symmetric matrix from (N, 3) vectors."""
    N = v.shape[0]
    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)
    x, y, z = v.unbind(1)
    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x
    return h


def _hat_inv(h):
    """Inverse of _hat: extract (N, 3) from skew-symmetric (N, 3, 3)."""
    return torch.stack([h[:, 2, 1], h[:, 0, 2], h[:, 1, 0]], dim=1)


def so3_exp_map(log_rot, eps=0.0001):
    """Axis-angle (N, 3) → rotation matrices (N, 3, 3) via Rodrigues."""
    nrms_sq = (log_rot * log_rot).sum(1)
    angles = torch.clamp(nrms_sq, min=eps * eps).sqrt()
    angles_inv = 1.0 / angles
    fac1 = angles_inv * angles.sin()
    fac2 = angles_inv * angles_inv * (1.0 - angles.cos())
    skews = _hat(log_rot)
    skews_sq = torch.bmm(skews, skews)
    eye = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device).unsqueeze(0)
    return fac1[:, None, None] * skews + fac2[:, None, None] * skews_sq + eye


def _acos_linear_extrapolation(x, bounds=(-(1.0 - 1e-4), 1.0 - 1e-4)):
    """acos with linear extrapolation near ±1 to avoid NaN gradients."""
    lower, upper = bounds
    clamped = x.clamp(lower, upper)
    return torch.acos(clamped)


def so3_log_map(R, eps=0.0001):
    """Rotation matrices (N, 3, 3) → axis-angle (N, 3) via matrix log."""
    phi = _acos_linear_extrapolation(
        0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0)
    )
    phi_sin = phi.sin()
    phi_factor = torch.empty_like(phi)
    small = phi.abs() < eps
    big = ~small
    if small.any():
        phi_factor[small] = 0.5 + (phi[small] ** 2) / 12.0
    if big.any():
        phi_factor[big] = phi[big] / (2.0 * phi_sin[big])
    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))
    return _hat_inv(log_rot_hat)


# ── euler_angles_to_matrix ─────────────────────────────────────────────


def _axis_rotation(axis, angle):
    """Single-axis rotation matrix. axis is 'X', 'Y', or 'Z'."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        rows = [
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, cos, -sin], dim=-1),
            torch.stack([zero, sin, cos], dim=-1),
        ]
    elif axis == "Y":
        rows = [
            torch.stack([cos, zero, sin], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sin, zero, cos], dim=-1),
        ]
    elif axis == "Z":
        rows = [
            torch.stack([cos, -sin, zero], dim=-1),
            torch.stack([sin, cos, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ]
    else:
        raise ValueError(f"Invalid axis: {axis}")
    return torch.stack(rows, dim=-2)


def euler_angles_to_matrix(euler_angles, convention):
    """Euler angles (..., 3) → rotation matrices (..., 3, 3)."""
    if len(convention) != 3 or not all(c in "XYZ" for c in convention):
        raise ValueError(f"Invalid convention: {convention}")
    matrices = [
        _axis_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return matrices[0] @ matrices[1] @ matrices[2]

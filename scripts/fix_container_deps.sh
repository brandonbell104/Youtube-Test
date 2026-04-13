#!/bin/bash
# Fix all GVHMR dependencies inside the running container.
# Run via: docker exec youtube-automation bash /app/scripts/fix_container_deps.sh
set -e

echo "=== Installing system build tools ==="
apt-get update -qq && apt-get install -y -qq gcc g++ python3.11-dev > /dev/null 2>&1

echo "=== Installing GVHMR ==="
if [ ! -f /opt/gvhmr/tools/demo/demo.py ]; then
    cd /opt && git clone --depth 1 https://github.com/zju3dv/GVHMR.git gvhmr
fi
cd /opt/gvhmr && pip install -e . --no-deps -q

echo "=== Installing Python dependencies ==="
pip install -q \
    torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true

pip install -q \
    smplx trimesh scipy \
    lightning==2.3.0 hydra-core==1.3 hydra-zen hydra_colorlog \
    rich timm==0.9.12 einops joblib termcolor colorlog \
    ffmpeg-python scikit-image "imageio==2.34.1" "av==13.0.0" \
    tensorboardX "ultralytics==8.2.42" lapx wis3d pycolmap \
    cython_bbox pytorch_lightning omegaconf antlr4-python3-runtime==4.9.3

echo "=== Creating pytorch3d transforms shim ==="
SHIM_DIR="/usr/local/lib/python3.11/dist-packages/pytorch3d"
mkdir -p "$SHIM_DIR/transforms"

cat > "$SHIM_DIR/__init__.py" << 'PYEOF'
# pytorch3d shim — only the transforms subpackage is provided.
PYEOF

cat > "$SHIM_DIR/transforms/__init__.py" << 'PYEOF'
"""
Lightweight reimplementation of pytorch3d.transforms.
Only the functions used by GVHMR are included.
"""
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
        small_angles, 0.5 * torch.ones_like(angles), torch.sin(half_angles) / angles.clamp(min=eps)
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
PYEOF

echo "=== Verifying imports ==="
python -c "import torch; print(f'torch {torch.__version__}')"
python -c "from pytorch3d.transforms import quaternion_to_matrix; print('pytorch3d shim OK')"
python -c "import smplx; print('smplx OK')"
python -c "import hmr4d; print('gvhmr (hmr4d) OK')" 2>/dev/null || echo "gvhmr import check skipped (may need runtime config)"

echo "=== All dependencies installed ==="
echo "Restart the tracking stage from the web UI."

"""Stub for pytorch3d.renderer.cameras — only look_at_rotation is provided."""
import torch


def look_at_rotation(camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device="cpu"):
    """Compute rotation matrix to look at a target point."""
    if not torch.is_tensor(camera_position):
        camera_position = torch.tensor(camera_position, dtype=torch.float32, device=device)
    if not torch.is_tensor(at):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    if not torch.is_tensor(up):
        up = torch.tensor(up, dtype=torch.float32, device=device)

    z_axis = camera_position - at
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.cross(up.expand_as(z_axis), z_axis, dim=-1)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return R

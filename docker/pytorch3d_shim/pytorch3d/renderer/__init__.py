"""
pytorch3d.renderer stub — provides importable placeholders so GVHMR modules
that import renderer symbols at the top level don't crash. These are only
used for visualization, not inference.
"""
from .cameras import look_at_rotation


class _Stub:
    def __init__(self, *a, **kw):
        raise NotImplementedError("pytorch3d.renderer is not available (shim only)")

    def __call__(self, *a, **kw):
        raise NotImplementedError("pytorch3d.renderer is not available (shim only)")


RasterizationSettings = _Stub
MeshRenderer = _Stub
MeshRasterizer = _Stub
SoftPhongShader = _Stub
HardPhongShader = _Stub
TexturesVertex = _Stub
PointLights = _Stub
PerspectiveCameras = _Stub

"""
Runtime verification of the SMPL → Blender retargeting in
src/blender_scripts/render.py.

Runs inside Blender's Python. Either:
    blender --background --python scripts/test_blender_retarget.py
or with the `bpy` PyPI module:
    python scripts/test_blender_retarget.py

What it does:
  1. Generates a synthetic SMPL-like skeleton (24 joints, real SMPL
     kinematic tree) plus random pose parameters for several frames.
  2. Builds the armature with render.py's build_smpl_avatar() and applies
     the poses with render.py's apply_tracking() — the exact production
     code path.
  3. Independently computes every joint's world position with a numpy
     implementation of SMPL's forward kinematics (rigid chain
     A_i = A_parent · [R_i | j_i - j_parent]).
  4. Asserts Blender's evaluated pose-bone positions match the reference
     to within 1e-4 m on every joint of every frame.
  5. Verifies fps/time-scale keyframe resampling places keys where expected.

Exits non-zero on failure.
"""

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

import bpy
from mathutils import Matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RENDER_PY = PROJECT_ROOT / "src" / "blender_scripts" / "render.py"

# SMPL kinematic tree (parent index per joint, pelvis = -1)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Plausible SMPL rest-pose joint positions (Y-up, meters)
REST_JOINTS = np.array([
    [0.00, 0.94, 0.00],   # pelvis
    [0.07, 0.87, 0.00],   # left_hip
    [-0.07, 0.87, 0.00],  # right_hip
    [0.00, 1.05, 0.01],   # spine1
    [0.10, 0.50, 0.00],   # left_knee
    [-0.10, 0.50, 0.00],  # right_knee
    [0.00, 1.16, 0.00],   # spine2
    [0.10, 0.09, 0.00],   # left_ankle
    [-0.10, 0.09, 0.00],  # right_ankle
    [0.00, 1.27, 0.01],   # spine3
    [0.11, 0.02, 0.12],   # left_foot
    [-0.11, 0.02, 0.12],  # right_foot
    [0.00, 1.42, 0.00],   # neck
    [0.07, 1.38, 0.00],   # left_collar
    [-0.07, 1.38, 0.00],  # right_collar
    [0.00, 1.51, 0.03],   # head
    [0.17, 1.40, 0.00],   # left_shoulder
    [-0.17, 1.40, 0.00],  # right_shoulder
    [0.42, 1.39, 0.00],   # left_elbow
    [-0.42, 1.39, 0.00],  # right_elbow
    [0.67, 1.38, 0.00],   # left_wrist
    [-0.67, 1.38, 0.00],  # right_wrist
    [0.75, 1.38, 0.00],   # left_hand
    [-0.75, 1.38, 0.00],  # right_hand
], dtype=np.float64)

NUM_FRAMES = 5
TOLERANCE = 1e-4


def load_render_module():
    spec = importlib.util.spec_from_file_location("pipeline_render", RENDER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rodrigues(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) → rotation matrix (3,3)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.eye(3)
    axis = aa / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def reference_fk(global_orient, body_pose, transl):
    """SMPL forward kinematics → world joint positions (24, 3), Y-up space."""
    rots = [rodrigues(np.asarray(global_orient, dtype=np.float64))]
    for j in range(1, 24):
        if j <= 21:
            aa = np.asarray(body_pose[(j - 1) * 3:(j - 1) * 3 + 3], dtype=np.float64)
            rots.append(rodrigues(aa))
        else:
            rots.append(np.eye(3))  # hands: not in body_pose

    A = [None] * 24
    A[0] = np.eye(4)
    A[0][:3, :3] = rots[0]
    A[0][:3, 3] = REST_JOINTS[0]
    for i in range(1, 24):
        p = SMPL_PARENTS[i]
        local = np.eye(4)
        local[:3, :3] = rots[i]
        local[:3, 3] = REST_JOINTS[i] - REST_JOINTS[p]
        A[i] = A[p] @ local

    positions = np.stack([A[i][:3, 3] for i in range(24)])
    return positions + np.asarray(transl, dtype=np.float64)


def make_synthetic_data(tmp: Path, src_fps: float):
    rng = np.random.default_rng(42)
    frames = []
    for f in range(NUM_FRAMES):
        frames.append({
            "frame": f,
            "global_orient": (rng.uniform(-0.8, 0.8, 3)).tolist(),
            "body_pose": (rng.uniform(-0.7, 0.7, 63)).tolist(),
            "transl": (rng.uniform(-0.5, 0.5, 3)).tolist(),
        })

    render_mod = load_render_module()
    tracking = {
        "fps": src_fps,
        "total_frames": NUM_FRAMES,
        "joint_names": render_mod.SMPL_JOINT_NAMES,
        "betas": [0.0] * 10,
        "frames": frames,
    }
    rig = {
        "joint_names": render_mod.SMPL_JOINT_NAMES,
        "rest_joints": REST_JOINTS.tolist(),
        "parents": SMPL_PARENTS,
        "betas": [0.0] * 10,
    }
    (tmp / "tracking.json").write_text(json.dumps(tracking))
    (tmp / "smpl_rig.json").write_text(json.dumps(rig))

    # Tiny dummy mesh: one vertex per joint, fully weighted to that joint
    v = REST_JOINTS.astype(np.float32)
    faces = np.array([[i, (i + 1) % 24, (i + 2) % 24] for i in range(22)],
                     dtype=np.int32)
    weights = np.eye(24, dtype=np.float32)
    np.savez_compressed(tmp / "smpl_mesh.npz",
                        v_template=v, faces=faces, weights=weights)
    return render_mod, tracking


def check_pose_positions(render_mod, tracking):
    """Frames keyed 1:1 (src fps == render fps): compare joint positions."""
    render_mod.clear_scene()
    tmp = Path(tempfile.mkdtemp())
    # reuse the files written by caller's tmp via globals — rebuilt below

    armature = None
    tmpdir = Path(bpy.app.tempdir or tempfile.gettempdir())
    rig_path = tmpdir / "smpl_rig.json"
    mesh_path = tmpdir / "smpl_mesh.npz"
    rig = {
        "joint_names": render_mod.SMPL_JOINT_NAMES,
        "rest_joints": REST_JOINTS.tolist(),
        "parents": SMPL_PARENTS,
        "betas": [0.0] * 10,
    }
    rig_path.write_text(json.dumps(rig))
    v = REST_JOINTS.astype(np.float32)
    faces = np.array([[i, (i + 1) % 24, (i + 2) % 24] for i in range(22)],
                     dtype=np.int32)
    np.savez_compressed(mesh_path, v_template=v, faces=faces,
                        weights=np.eye(24, dtype=np.float32))

    armature = render_mod.build_smpl_avatar(str(rig_path), str(mesh_path))
    render_mod.apply_tracking(armature, tracking,
                              render_fps=tracking["fps"], time_scale=1.0)

    # SMPL Y-up → Blender Z-up conversion the armature object carries
    to_blender = Matrix.Rotation(math.radians(90), 4, "X")

    worst = 0.0
    for frame_data in tracking["frames"]:
        f = int(frame_data["frame"])
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()

        expected_smpl = reference_fk(
            frame_data["global_orient"],
            frame_data["body_pose"],
            frame_data["transl"],
        )
        for i, name in enumerate(render_mod.SMPL_JOINT_NAMES):
            pb = armature.pose.bones[name]
            actual = np.array(armature.matrix_world @ pb.head)
            expected = np.array(to_blender @ Matrix.Translation(expected_smpl[i]).translation)
            err = float(np.linalg.norm(actual - expected))
            worst = max(worst, err)
            if err > TOLERANCE:
                print(f"FAIL frame {f} joint {name}: "
                      f"blender={actual.round(5).tolist()} "
                      f"reference={expected.round(5).tolist()} err={err:.6f}")
                return False, worst
    return True, worst


def check_resampling(render_mod, tracking):
    """Verify keyframes land at frame * (render_fps/src_fps) * time_scale."""
    render_mod.clear_scene()
    tmpdir = Path(bpy.app.tempdir or tempfile.gettempdir())
    armature = render_mod.build_smpl_avatar(
        str(tmpdir / "smpl_rig.json"), str(tmpdir / "smpl_mesh.npz"))

    src_fps, render_fps, time_scale = 25.0, 30.0, 1.3
    tracking = dict(tracking, fps=src_fps)
    render_mod.apply_tracking(armature, tracking, render_fps, time_scale)

    scale = (render_fps / src_fps) * time_scale
    action = armature.animation_data.action
    fcu = next(fc for fc in action.fcurves
               if fc.data_path == 'pose.bones["left_knee"].rotation_quaternion')
    keyed = sorted(kp.co.x for kp in fcu.keyframe_points)
    expected = [f * scale for f in range(NUM_FRAMES)]
    ok = len(keyed) == NUM_FRAMES and all(
        abs(a - b) < 1e-3 for a, b in zip(keyed, expected))
    if not ok:
        print(f"FAIL resampling: keyed at {keyed}, expected {expected}")
    return ok


def main():
    print("=" * 60)
    print("SMPL → Blender retargeting verification")
    print(f"Blender {bpy.app.version_string}, {NUM_FRAMES} random frames, "
          f"tolerance {TOLERANCE} m")
    print("=" * 60)

    tmpdir = Path(bpy.app.tempdir or tempfile.gettempdir())
    render_mod, tracking = make_synthetic_data(tmpdir, src_fps=30.0)

    ok_pose, worst = check_pose_positions(render_mod, tracking)
    print(f"[1/2] Joint position check: "
          f"{'PASS' if ok_pose else 'FAIL'} (worst error {worst:.2e} m)")

    ok_resample = check_resampling(render_mod, tracking)
    print(f"[2/2] fps/time-scale resampling check: "
          f"{'PASS' if ok_resample else 'FAIL'}")

    if not (ok_pose and ok_resample):
        sys.exit(1)
    print("\nAll retargeting checks passed.")


if __name__ == "__main__":
    main()

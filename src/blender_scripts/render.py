"""
Blender headless render script.

Called via: blender --background --python render.py -- /path/to/render_config.json

This script:
  1. Builds an SMPL armature + skinned mesh from data pre-computed by the
     tracking stage (smpl_rig.json + smpl_mesh.npz). The tracking stage runs
     GVHMR to produce per-frame SMPL pose parameters.
  2. Applies the SMPL body_pose / global_orient / transl per frame as bone
     rotations and root location keyframes. Because SMPL outputs full joint
     rotations (including twist), no landmark-to-rotation reconstruction is
     needed — we set the bones directly.
  3. Applies lip sync viseme data to facial blend shapes (if available).
  4. Adds the voice audio track.
  5. Renders the final video with Cycles GPU.
"""

import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector, Euler, Matrix, Quaternion

import numpy as np


def load_config() -> dict:
    """Load render config from command line argument."""
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Usage: blender --background --python render.py -- config.json")
    args = argv[argv.index("--") + 1:]
    config_path = Path(args[0])
    return json.loads(config_path.read_text())


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


# ─────────────────────────────────────────────────────────────────────────
# SMPL rig + skinned mesh construction
# ─────────────────────────────────────────────────────────────────────────

# SMPL joint order — these 24 joint names become bone names in the armature.
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# NOTE on bone orientation: every bone is created pointing along +Y with
# zero roll, so each bone's rest-space axes coincide with the armature's
# axes. SMPL pose parameters are joint rotations expressed in frames that
# are axis-aligned with the body frame at rest, so with identity bone rest
# orientations the SMPL axis-angle rotations can be assigned directly to
# pose-bone quaternions and the kinematic chain composes exactly like
# SMPL's. Do NOT point bones at their children — that gives each bone an
# arbitrary rest rotation and scrambles the applied pose.


def build_smpl_avatar(rig_path: str, mesh_path: str) -> bpy.types.Object:
    """
    Build an SMPL armature + skinned mesh from the files exported by the
    tracking stage. Returns the armature object.

    rig_path:  smpl_rig.json  — rest joint positions + parents + betas
    mesh_path: smpl_mesh.npz  — v_template, faces, weights
    """
    rig = json.loads(Path(rig_path).read_text())
    joint_names = rig["joint_names"]  # 24 names
    rest_joints = np.array(rig["rest_joints"], dtype=np.float32)  # (24, 3)
    parents = rig["parents"]                                      # list of 24 ints

    npz = np.load(mesh_path)
    v_template = npz["v_template"]  # (6890, 3)
    faces = npz["faces"]             # (13776, 3)
    weights = npz["weights"]         # (6890, 24)

    print(f"Building SMPL avatar: {len(joint_names)} joints, "
          f"{len(v_template)} verts, {len(faces)} faces")

    # 1. Create armature
    arm_data = bpy.data.armatures.new("SMPL_Armature")
    armature = bpy.data.objects.new("SMPL_Avatar", arm_data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones

    # Create all bones with head at their rest joint position, tail along
    # +Y and zero roll so every bone's rest orientation is identity (see
    # note above — required for direct SMPL rotation assignment).
    for i, name in enumerate(joint_names):
        eb = edit_bones.new(name)
        head = Vector(rest_joints[i].tolist())
        eb.head = head
        eb.tail = head + Vector((0, 0.1, 0))
        eb.roll = 0.0

    # Wire up parents
    for i, name in enumerate(joint_names):
        p = parents[i]
        if p is not None and p >= 0:
            edit_bones[name].parent = edit_bones[joint_names[p]]
            # Don't connect — parent tail rarely coincides with child head
            edit_bones[name].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    # 2. Create mesh
    mesh = bpy.data.meshes.new("SMPL_Mesh")
    mesh_obj = bpy.data.objects.new("SMPL_Body", mesh)
    bpy.context.scene.collection.objects.link(mesh_obj)

    mesh.from_pydata(v_template.tolist(), [], faces.tolist())
    mesh.update()

    # 3. Vertex groups (one per joint)
    for name in joint_names:
        mesh_obj.vertex_groups.new(name=name)

    # 4. Assign skinning weights. Only write non-trivial weights to keep it fast.
    weight_threshold = 1e-4
    for v_idx in range(weights.shape[0]):
        w_row = weights[v_idx]
        for j_idx in range(weights.shape[1]):
            w = float(w_row[j_idx])
            if w > weight_threshold:
                mesh_obj.vertex_groups[joint_names[j_idx]].add([v_idx], w, "REPLACE")

    # 5. Parent mesh to armature with an armature modifier
    mesh_obj.parent = armature
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = armature
    mod.use_vertex_groups = True

    # 6. Simple material so the body isn't invisible in Cycles
    mat = bpy.data.materials.new(name="SMPL_Skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.85, 0.72, 0.62, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    mesh_obj.data.materials.append(mat)

    # 7. The rig/mesh data and all SMPL pose params live in SMPL's Y-up
    # coordinate system. Convert to Blender's Z-up world once, at the
    # object level, instead of per-keyframe.
    armature.rotation_euler = (math.radians(90), 0, 0)

    bpy.context.view_layer.objects.active = armature
    print("SMPL avatar built successfully")
    return armature


# ─────────────────────────────────────────────────────────────────────────
# SMPL pose application
# ─────────────────────────────────────────────────────────────────────────

def _axis_angle_to_quat(aa) -> Quaternion:
    """Convert a 3-vector axis-angle rotation to a Blender Quaternion."""
    x, y, z = float(aa[0]), float(aa[1]), float(aa[2])
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis = Vector((x / angle, y / angle, z / angle))
    return Quaternion(axis, angle)


def apply_tracking(armature: bpy.types.Object, tracking_data: dict, render_fps: float):
    """
    Apply per-frame SMPL pose parameters (from GVHMR) to the SMPL armature.

    Keyframes are resampled from the source video's fps (tracking_data["fps"])
    to the render fps, so motion plays back at real-time speed regardless of
    what fps the original video used. All rotations/translations are assigned
    in SMPL (Y-up) space directly — the armature object itself is rotated to
    Z-up in build_smpl_avatar(), and bones are built with identity rest
    orientation, so no per-bone coordinate conversion is needed.

    tracking_data schema (from src/stages/tracking.py):
      {
        "fps": float,
        "total_frames": int,
        "joint_names": [24 names],
        "betas": [10 floats],
        "frames": [
            {"frame": int, "global_orient": [3], "body_pose": [63], "transl": [3]},
            ...
        ]
      }
    """
    frames = tracking_data["frames"]
    joint_names = tracking_data.get("joint_names", SMPL_JOINT_NAMES)
    total = len(frames)
    src_fps = float(tracking_data.get("fps") or render_fps)
    frame_scale = render_fps / src_fps

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    for pb in armature.pose.bones:
        pb.rotation_mode = "QUATERNION"

    pelvis_bone = armature.pose.bones.get(joint_names[0])
    if pelvis_bone is None:
        raise RuntimeError(f"Pelvis bone '{joint_names[0]}' not found in armature")

    tracked = 0
    for frame_data in frames:
        # Resample: source frame index → render timeline frame (float ok)
        out_frame = float(frame_data["frame"]) * frame_scale

        global_orient = frame_data["global_orient"]       # (3,) axis-angle
        body_pose_flat = frame_data["body_pose"]          # (63,) 21 joints * 3
        transl = frame_data["transl"]                     # (3,) world translation

        # Pelvis: global orientation + translation, both in SMPL space.
        pelvis_bone.rotation_quaternion = _axis_angle_to_quat(global_orient)
        pelvis_bone.keyframe_insert(data_path="rotation_quaternion", frame=out_frame)

        pelvis_bone.location = Vector((float(transl[0]), float(transl[1]), float(transl[2])))
        pelvis_bone.keyframe_insert(data_path="location", frame=out_frame)

        # body_pose is 63-dim = 21 joints (indices 1..21 in SMPL) × 3
        # Joint indices 22 and 23 are hands, not included in body_pose.
        for j in range(1, 22):
            aa = body_pose_flat[(j - 1) * 3:(j - 1) * 3 + 3]
            bone = armature.pose.bones.get(joint_names[j])
            if bone is None:
                continue
            bone.rotation_quaternion = _axis_angle_to_quat(aa)
            bone.keyframe_insert(data_path="rotation_quaternion", frame=out_frame)

        tracked += 1
        if tracked % 200 == 0:
            print(f"Applied SMPL pose: {tracked}/{total} frames")

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"Applied SMPL pose to {tracked} frames "
          f"(src {src_fps:.2f} fps → render {render_fps:.2f} fps)")


def apply_lipsync(armature: bpy.types.Object, lipsync_data: dict, fps: float):
    """Apply lip sync viseme data to facial blend shapes."""
    cues = lipsync_data.get("cues", [])
    if not cues:
        print("No lip sync cues to apply")
        return

    # Find mesh objects with shape keys (blend shapes)
    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.data.shape_keys:
            mesh_obj = obj
            break

    if mesh_obj is None:
        print("Warning: No mesh with shape keys found — skipping lip sync")
        return

    shape_keys = mesh_obj.data.shape_keys.key_blocks
    available_shapes = {sk.name.lower(): sk.name for sk in shape_keys}

    print(f"Available shape keys: {list(available_shapes.values())}")

    for cue in cues:
        blend_shape = cue["blend_shape"]
        start_frame = int(cue["start"] * fps)
        end_frame = int(cue["end"] * fps)

        # Find matching shape key (case-insensitive)
        shape_name = available_shapes.get(blend_shape.lower())
        if shape_name is None:
            # Try without prefix
            short_name = blend_shape.replace("viseme_", "")
            shape_name = available_shapes.get(short_name.lower())

        if shape_name is None:
            continue

        sk = shape_keys[shape_name]

        # Set keyframes: ramp up → hold → ramp down
        transition = max(1, int(fps * 0.03))  # ~1 frame transition

        # Before: value = 0
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=max(0, start_frame - transition))

        # Peak: value = 1
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=start_frame)
        sk.keyframe_insert(data_path="value", frame=end_frame)

        # After: value = 0
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=end_frame + transition)

    print(f"Applied {len(cues)} lip sync cues")


def setup_camera_and_lighting():
    """Set up a basic camera and lighting setup for the avatar."""
    # Camera
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = (0, -3, 1.5)
    cam_obj.rotation_euler = (math.radians(80), 0, 0)
    bpy.context.scene.camera = cam_obj

    # Key light
    key_light = bpy.data.lights.new("KeyLight", "AREA")
    key_light.energy = 500
    key_light.size = 3
    key_obj = bpy.data.objects.new("KeyLight", key_light)
    bpy.context.scene.collection.objects.link(key_obj)
    key_obj.location = (2, -2, 3)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(30))

    # Fill light
    fill_light = bpy.data.lights.new("FillLight", "AREA")
    fill_light.energy = 200
    fill_light.size = 2
    fill_obj = bpy.data.objects.new("FillLight", fill_light)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (-2, -2, 2)

    # Background
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.15, 0.15, 0.18, 1.0)  # Dark gray


def setup_render_settings(config: dict):
    """Configure render settings."""
    scene = bpy.context.scene

    # Use Cycles with CUDA GPU rendering
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for device in prefs.devices:
        if device.type == "CUDA":
            device.use = True
            print(f"  GPU enabled: {device.name}")
        else:
            device.use = False
    scene.cycles.device = "GPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False  # Denoising runs on CPU and is the bottleneck
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.1
    print("Using Cycles GPU (CUDA) rendering — 16 samples, no denoising")
    scene.render.resolution_x = config["resolution_x"]
    scene.render.resolution_y = config["resolution_y"]
    scene.render.fps = config["fps"]

    # Output settings
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.audio_codec = "AAC"
    scene.render.ffmpeg.audio_bitrate = 192

    # EEVEE settings for quality
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = True
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 64


def add_audio(voice_path: str):
    """Add the voice audio as a sound strip in the sequencer."""
    scene = bpy.context.scene
    if not scene.sequence_editor:
        scene.sequence_editor_create()

    seq = scene.sequence_editor
    # Start at frame 0 to match scene.frame_start / the first pose keyframe
    sound_strip = seq.sequences.new_sound(
        name="Voice",
        filepath=voice_path,
        channel=1,
        frame_start=0,
    )

    # Adjust scene length to match audio
    audio_frames = sound_strip.frame_final_end
    if audio_frames > scene.frame_end:
        scene.frame_end = audio_frames


def main():
    config = load_config()

    print("=" * 60)
    print("YouTube Automation — Blender Render")
    print("=" * 60)

    # Clean slate
    clear_scene()

    # Build SMPL avatar from data exported by the tracking stage.
    # The tracking stage writes smpl_rig.json + smpl_mesh.npz alongside
    # tracking.json in the tracking stage output directory.
    tracking_path = Path(config["tracking_path"])
    tracking_dir = tracking_path.parent
    rig_path = tracking_dir / "smpl_rig.json"
    mesh_path = tracking_dir / "smpl_mesh.npz"

    if not rig_path.exists() or not mesh_path.exists():
        raise RuntimeError(
            f"SMPL rig/mesh data not found in {tracking_dir}. "
            "Re-run the tracking stage with the GVHMR-based tracker."
        )

    print(f"Building SMPL avatar from: {rig_path.name} + {mesh_path.name}")
    armature = build_smpl_avatar(str(rig_path), str(mesh_path))

    # Setup camera and lighting
    setup_camera_and_lighting()

    # Load and apply tracking data (SMPL pose params per frame),
    # resampled from the source video fps to the render fps.
    print(f"Loading tracking data: {tracking_path}")
    tracking_data = json.loads(tracking_path.read_text())
    render_fps = float(config["fps"])
    apply_tracking(armature, tracking_data, render_fps)

    # Load and apply lip sync
    print(f"Loading lip sync data: {config['lipsync_path']}")
    lipsync_data = json.loads(Path(config["lipsync_path"]).read_text())
    apply_lipsync(armature, lipsync_data, float(config["fps"]))

    # Add audio
    print(f"Adding audio: {config['voice_path']}")
    add_audio(config["voice_path"])

    # Configure render
    setup_render_settings(config)

    # Set frame range — tracking frames are resampled to the render fps,
    # so scale the end frame the same way apply_tracking scales keyframes.
    bpy.context.scene.frame_start = 0
    total_frames = tracking_data.get("total_frames", 300)
    src_fps = float(tracking_data.get("fps") or render_fps)
    motion_end = int(math.ceil(total_frames * render_fps / src_fps))
    bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, motion_end)

    # Render
    output_path = config["output_path"]
    bpy.context.scene.render.filepath = output_path
    print(f"Rendering {total_frames} frames to {output_path}")
    bpy.ops.render.render(animation=True)

    print("Render complete!")


if __name__ == "__main__":
    main()

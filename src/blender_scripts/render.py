"""
Blender headless render script.

Called via: blender --background --python render.py -- /path/to/render_config.json

This script:
  1. Imports a 3D avatar (.glb/.fbx)
  2. Applies body pose tracking data from MediaPipe to the armature
  3. Applies lip sync viseme data to facial blend shapes
  4. Adds the voice audio track
  5. Renders the final video
"""

import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector, Euler, Matrix, Quaternion


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


def import_avatar(avatar_path: str) -> bpy.types.Object:
    """Import a 3D avatar and return the armature object."""
    path = Path(avatar_path)
    ext = path.suffix.lower()

    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"Unsupported avatar format: {ext}")

    # Find the armature
    armature = None
    for obj in bpy.context.selected_objects:
        if obj.type == "ARMATURE":
            armature = obj
            break

    if armature is None:
        # Look in all scene objects
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                armature = obj
                break

    if armature is None:
        # Some glTF models import bones as Empty objects — print what we got
        print("WARNING: No armature found. Objects in scene:")
        for obj in bpy.data.objects:
            print(f"  - {obj.name} (type={obj.type})")

        # Try to find any mesh and proceed without armature
        # The render will still work, just without pose animation
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                print(f"Using mesh object '{obj.name}' without armature")
                return None

        raise RuntimeError("No armature or mesh found in imported avatar")

    return armature


# MediaPipe landmark → common bone name mapping
# This maps MediaPipe pose landmarks to typical armature bone names
# Adjust these mappings based on your specific avatar rig
LANDMARK_TO_BONE = {
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "nose": "Head",
}

# Alternative bone name formats (Mixamo, Rigify, etc.)
BONE_NAME_VARIANTS = {
    "LeftArm": ["LeftArm", "mixamorig:LeftArm", "Left_Arm", "upper_arm.L", "arm.L"],
    "RightArm": ["RightArm", "mixamorig:RightArm", "Right_Arm", "upper_arm.R", "arm.R"],
    "LeftForeArm": ["LeftForeArm", "mixamorig:LeftForeArm", "Left_ForeArm", "forearm.L"],
    "RightForeArm": ["RightForeArm", "mixamorig:RightForeArm", "Right_ForeArm", "forearm.R"],
    "LeftHand": ["LeftHand", "mixamorig:LeftHand", "Left_Hand", "hand.L"],
    "RightHand": ["RightHand", "mixamorig:RightHand", "Right_Hand", "hand.R"],
    "LeftUpLeg": ["LeftUpLeg", "mixamorig:LeftUpLeg", "Left_UpLeg", "thigh.L", "upper_leg.L"],
    "RightUpLeg": ["RightUpLeg", "mixamorig:RightUpLeg", "Right_UpLeg", "thigh.R", "upper_leg.R"],
    "LeftLeg": ["LeftLeg", "mixamorig:LeftLeg", "Left_Leg", "shin.L", "lower_leg.L"],
    "RightLeg": ["RightLeg", "mixamorig:RightLeg", "Right_Leg", "shin.R", "lower_leg.R"],
    "LeftFoot": ["LeftFoot", "mixamorig:LeftFoot", "Left_Foot", "foot.L"],
    "RightFoot": ["RightFoot", "mixamorig:RightFoot", "Right_Foot", "foot.R"],
    "Head": ["Head", "mixamorig:Head", "head"],
    "Hips": ["Hips", "mixamorig:Hips", "hips", "root", "Root"],
    "Spine": ["Spine", "mixamorig:Spine", "spine", "Spine1"],
}


def find_bone(armature: bpy.types.Object, canonical_name: str) -> str | None:
    """Find a bone by trying multiple naming conventions."""
    variants = BONE_NAME_VARIANTS.get(canonical_name, [canonical_name])
    bone_names = [b.name for b in armature.data.bones]

    for variant in variants:
        if variant in bone_names:
            return variant

    # Case-insensitive fallback
    for variant in variants:
        for bn in bone_names:
            if bn.lower() == variant.lower():
                return bn

    return None


def mediapipe_to_blender(lm: dict) -> Vector:
    """Convert a MediaPipe landmark to Blender coordinate space.
    MediaPipe: X=right, Y=down, Z=toward camera
    Blender:   X=right, Y=forward (into screen), Z=up
    """
    return Vector((lm["x"], -lm["z"], -lm["y"]))


def compute_pose_rotation(pose_bone, target_direction: Vector):
    """
    Compute the pose bone quaternion that points the bone in target_direction
    (given in armature space), properly converted to bone-local space.
    """
    bone = pose_bone.bone

    # Bone rest direction in armature space
    rest_dir = (bone.tail_local - bone.head_local).normalized()

    if target_direction.length < 0.001:
        return None

    # Delta rotation from rest direction to target direction (in armature space)
    armature_delta = rest_dir.rotation_difference(target_direction)

    # Target bone matrix in armature space = delta applied to rest matrix
    rest_mat = bone.matrix_local
    target_mat = armature_delta.to_matrix().to_4x4() @ rest_mat

    # Convert both to bone-local space (relative to parent rest pose)
    if bone.parent:
        parent_inv = bone.parent.matrix_local.inverted()
        local_target = parent_inv @ target_mat
        local_rest = parent_inv @ rest_mat
    else:
        local_target = target_mat
        local_rest = rest_mat

    # Pose rotation = rest_local^-1 @ target_local
    pose_rot = local_rest.to_quaternion().inverted() @ local_target.to_quaternion()
    return pose_rot


def apply_tracking(armature: bpy.types.Object, tracking_data: dict):
    """Apply MediaPipe body tracking data to the armature as keyframes."""
    frames = tracking_data["frames"]
    fps = tracking_data["fps"]

    # Enter pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # Set rotation mode to quaternion for all pose bones
    for pb in armature.pose.bones:
        pb.rotation_mode = "QUATERNION"

    # Map bones
    bone_map = {}
    for canonical in BONE_NAME_VARIANTS:
        actual = find_bone(armature, canonical)
        if actual:
            bone_map[canonical] = actual

    print(f"Mapped {len(bone_map)} bones: {list(bone_map.keys())}")

    # Print bone hierarchy for debugging
    for canonical, actual_name in bone_map.items():
        bone = armature.data.bones.get(actual_name)
        if bone:
            parent_name = bone.parent.name if bone.parent else "None"
            rest_dir = (bone.tail_local - bone.head_local).normalized()
            print(f"  {canonical} -> {actual_name} (parent: {parent_name}, rest_dir: {rest_dir})")

    # Limb pairs: (parent_landmark, child_landmark, bone_canonical)
    limb_pairs = [
        ("left_shoulder", "left_elbow", "LeftArm"),
        ("left_elbow", "left_wrist", "LeftForeArm"),
        ("right_shoulder", "right_elbow", "RightArm"),
        ("right_elbow", "right_wrist", "RightForeArm"),
        ("left_hip", "left_knee", "LeftUpLeg"),
        ("left_knee", "left_ankle", "LeftLeg"),
        ("right_hip", "right_knee", "RightUpLeg"),
        ("right_knee", "right_ankle", "RightLeg"),
    ]

    # Spine tracking: use shoulders midpoint → nose direction
    spine_pairs = [
        ("Spine", "left_shoulder", "right_shoulder", "nose"),
    ]

    tracked_frame_count = 0
    # For smoothing: store previous rotations per bone
    prev_rotations = {}
    smooth_factor = 0.3  # 0 = no smoothing, 1 = completely frozen

    # Process each frame
    for frame_data in frames:
        frame_idx = frame_data["frame"]
        landmarks = frame_data.get("world_landmarks") or frame_data.get("landmarks")

        if landmarks is None:
            continue

        # Build landmark dict by name
        lm_dict = {lm["name"]: lm for lm in landmarks}

        bpy.context.scene.frame_set(frame_idx)

        # Apply root (hip) position
        hips_bone_name = bone_map.get("Hips")
        if hips_bone_name and "left_hip" in lm_dict and "right_hip" in lm_dict:
            pose_bone = armature.pose.bones.get(hips_bone_name)
            if pose_bone:
                lh = mediapipe_to_blender(lm_dict["left_hip"])
                rh = mediapipe_to_blender(lm_dict["right_hip"])
                scale = 2.0
                hip_center = (lh + rh) / 2 * scale
                pose_bone.location = hip_center
                pose_bone.keyframe_insert(data_path="location", frame=frame_idx)

        # Apply limb rotations
        for parent_lm, child_lm, bone_canonical in limb_pairs:
            if parent_lm not in lm_dict or child_lm not in lm_dict:
                continue

            actual_name = bone_map.get(bone_canonical)
            if not actual_name:
                continue

            pose_bone = armature.pose.bones.get(actual_name)
            if not pose_bone:
                continue

            # Compute target direction in Blender armature space
            origin = mediapipe_to_blender(lm_dict[parent_lm])
            target = mediapipe_to_blender(lm_dict[child_lm])
            direction = (target - origin).normalized()

            rotation = compute_pose_rotation(pose_bone, direction)
            if rotation is not None:
                # Apply smoothing to reduce jitter
                if bone_canonical in prev_rotations:
                    rotation = prev_rotations[bone_canonical].slerp(rotation, 1.0 - smooth_factor)
                prev_rotations[bone_canonical] = rotation.copy()

                pose_bone.rotation_quaternion = rotation
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

        # Apply spine rotation (shoulders midpoint toward nose)
        for spine_bone, lm_left, lm_right, lm_target in spine_pairs:
            if lm_left not in lm_dict or lm_right not in lm_dict or lm_target not in lm_dict:
                continue
            actual_name = bone_map.get(spine_bone)
            if not actual_name:
                continue
            pose_bone = armature.pose.bones.get(actual_name)
            if not pose_bone:
                continue

            left = mediapipe_to_blender(lm_dict[lm_left])
            right = mediapipe_to_blender(lm_dict[lm_right])
            head = mediapipe_to_blender(lm_dict[lm_target])
            mid_shoulder = (left + right) / 2
            direction = (head - mid_shoulder).normalized()

            rotation = compute_pose_rotation(pose_bone, direction)
            if rotation is not None:
                if spine_bone in prev_rotations:
                    rotation = prev_rotations[spine_bone].slerp(rotation, 1.0 - smooth_factor)
                prev_rotations[spine_bone] = rotation.copy()

                pose_bone.rotation_quaternion = rotation
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

        tracked_frame_count += 1

        if tracked_frame_count % 200 == 0:
            print(f"Applied tracking: {tracked_frame_count} frames processed")

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"Applied tracking to {tracked_frame_count} frames")


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
    sound_strip = seq.sequences.new_sound(
        name="Voice",
        filepath=voice_path,
        channel=1,
        frame_start=1,
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

    # Import avatar
    print(f"Importing avatar: {config['avatar_path']}")
    armature = import_avatar(config["avatar_path"])

    # Setup camera and lighting
    setup_camera_and_lighting()

    # Load and apply tracking data
    print(f"Loading tracking data: {config['tracking_path']}")
    tracking_data = json.loads(Path(config["tracking_path"]).read_text())
    if armature is not None:
        apply_tracking(armature, tracking_data)
    else:
        print("Skipping tracking — no armature available")

    # Load and apply lip sync
    print(f"Loading lip sync data: {config['lipsync_path']}")
    lipsync_data = json.loads(Path(config["lipsync_path"]).read_text())
    if armature is not None:
        apply_lipsync(armature, lipsync_data, float(config["fps"]))
    else:
        print("Skipping lip sync — no armature available")

    # Add audio
    print(f"Adding audio: {config['voice_path']}")
    add_audio(config["voice_path"])

    # Configure render
    setup_render_settings(config)

    # Set frame range
    bpy.context.scene.frame_start = 0
    total_frames = tracking_data.get("total_frames", 300)
    bpy.context.scene.frame_end = total_frames

    # Render
    output_path = config["output_path"]
    bpy.context.scene.render.filepath = output_path
    print(f"Rendering {total_frames} frames to {output_path}")
    bpy.ops.render.render(animation=True)

    print("Render complete!")


if __name__ == "__main__":
    main()

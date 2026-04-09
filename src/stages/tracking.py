"""
Stage 5: 3D body pose tracking using GVHMR (SMPL-based).

GVHMR (SIGGRAPH Asia 2024) regresses SMPL parameters directly from monocular
video, outputting per-frame joint rotations in axis-angle form — unlike
MediaPipe, which only gives landmark positions. SMPL rotations include
bone twist, are trained on mocap data, and are gravity-aligned.

Pipeline:
    1. Run GVHMR demo script on the downloaded video → per-frame SMPL params.
    2. Evaluate the SMPL body model at every frame using `smplx` to bake out
       per-frame joint rotations in world/global orientation form.
    3. Also extract SMPL rest-pose data (joint positions, kinematic tree,
       vertex template, skinning weights) so the Blender render stage can
       build an armature + skinned mesh without needing torch/smplx inside
       Blender's bundled Python.

Outputs (written to the stage output directory):
    tracking.json      — fps, total frames, per-frame SMPL pose parameters
                         (body_pose 63-d, global_orient 3-d, transl 3-d)
    smpl_rig.json      — rest joint positions, parent indices, betas
    smpl_mesh.npz      — v_template (6890,3), faces (13776,3),
                         weights (6890,24). Consumed by render.py in Blender.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


# SMPL joint name order (indices 0-23). Used for debugging and mapping.
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


class TrackingStage(Stage):
    name = "tracking"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        video_path = self.prev_stage_dir(job_id, "download") / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        smpl_pkl = self.config.smpl_model_dir / "SMPL_NEUTRAL.pkl"
        if not smpl_pkl.exists():
            raise FileNotFoundError(
                f"SMPL body model not found at {smpl_pkl}. "
                "Register at https://smpl.is.tue.mpg.de, download "
                "SMPL_python_v.1.1.0, extract SMPL_NEUTRAL.pkl, "
                f"and place it at {smpl_pkl}"
            )

        # Step 1: Run GVHMR demo on the video
        smpl_params = self._run_gvhmr(video_path, out_dir)

        # Step 2: Load SMPL model and prepare rig/mesh data for Blender
        self._export_smpl_rig_and_mesh(smpl_params, out_dir)

        # Step 3: Write tracking.json (per-frame SMPL pose params)
        fps = smpl_params["fps"]
        total_frames = smpl_params["num_frames"]
        tracking_output = {
            "tracker": "gvhmr",
            "smpl_gender": self.config.smpl_gender,
            "fps": fps,
            "total_frames": total_frames,
            "tracked_frames": total_frames,
            "joint_names": SMPL_JOINT_NAMES,
            "betas": smpl_params["betas"].tolist(),  # (10,)
            "frames": [
                {
                    "frame": i,
                    "global_orient": smpl_params["global_orient"][i].tolist(),  # (3,)
                    "body_pose": smpl_params["body_pose"][i].tolist(),          # (63,)
                    "transl": smpl_params["transl"][i].tolist(),                # (3,)
                }
                for i in range(total_frames)
            ],
        }

        (out_dir / "tracking.json").write_text(
            json.dumps(tracking_output, separators=(",", ":"))
        )

        logger.info(
            "Tracking complete: %d frames, fps=%.1f, SMPL params saved",
            total_frames, fps,
        )

        return {
            "tracker": "gvhmr",
            "total_frames": total_frames,
            "tracked_frames": total_frames,
            "fps": fps,
            "tracking_rate": 100.0,
        }

    # ─────────────────────────────────────────────────────────────────────
    # GVHMR execution
    # ─────────────────────────────────────────────────────────────────────

    def _run_gvhmr(self, video_path: Path, out_dir: Path) -> dict:
        """Run GVHMR on the video and return per-frame SMPL parameters."""
        import cv2

        # Probe fps
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        gvhmr_root = self.config.gvhmr_root
        demo_script = gvhmr_root / "tools" / "demo" / "demo.py"
        if not demo_script.exists():
            raise FileNotFoundError(
                f"GVHMR demo script not found at {demo_script}. "
                "Check that GVHMR_ROOT is set correctly in the container."
            )

        gvhmr_output_root = out_dir / "gvhmr_out"
        gvhmr_output_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", str(demo_script),
            f"--video={video_path}",
            f"--output_root={gvhmr_output_root}",
        ]
        if self.config.tracking_static_camera:
            cmd.append("-s")

        logger.info("Running GVHMR: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=str(gvhmr_root),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode != 0:
            logger.error("GVHMR stdout: %s", proc.stdout[-2000:])
            logger.error("GVHMR stderr: %s", proc.stderr[-2000:])
            raise RuntimeError(f"GVHMR failed with code {proc.returncode}")
        logger.info("GVHMR finished")

        # GVHMR saves results to a subdirectory named after the video basename.
        # The predictions file is cfg.paths.hmr4d_results; find it.
        candidates = list(gvhmr_output_root.rglob("hmr4d_results.pt"))
        if not candidates:
            # Fallback: search for any .pt file
            candidates = list(gvhmr_output_root.rglob("*.pt"))
        if not candidates:
            raise RuntimeError(
                f"No GVHMR predictions file found under {gvhmr_output_root}"
            )
        results_pt = candidates[0]
        logger.info("Loading GVHMR predictions from %s", results_pt)

        import torch
        pred = torch.load(str(results_pt), map_location="cpu", weights_only=False)

        # Prefer the gravity-aligned global parameters; fall back to incam.
        smpl_params = pred.get("smpl_params_global") or pred.get("smpl_params_incam")
        if smpl_params is None:
            raise RuntimeError(
                f"GVHMR output missing smpl_params_global/incam. Keys: {list(pred.keys())}"
            )

        def _to_numpy(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        body_pose = _to_numpy(smpl_params["body_pose"])          # (T, 63) expected
        global_orient = _to_numpy(smpl_params["global_orient"])   # (T, 3)
        transl = _to_numpy(smpl_params["transl"])                 # (T, 3)
        betas = _to_numpy(smpl_params["betas"])                   # (T, 10) or (10,)

        # Squeeze any leading batch dim
        for name, arr in (("body_pose", body_pose), ("global_orient", global_orient),
                          ("transl", transl), ("betas", betas)):
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if name == "betas" and arr.ndim == 2:
                # (T, 10) → take mean across frames as a stable shape
                arr = arr.mean(axis=0)
            if name == "body_pose":
                body_pose = arr
            elif name == "global_orient":
                global_orient = arr
            elif name == "transl":
                transl = arr
            elif name == "betas":
                betas = arr

        num_frames = body_pose.shape[0]
        # Sanity: body_pose is 63-d (21 joints * 3) per frame
        if body_pose.shape[-1] == 69:
            # SMPL full body_pose is 23 joints × 3 = 69; strip hands if needed
            body_pose = body_pose[..., :63]
        assert body_pose.shape[-1] == 63, f"Unexpected body_pose shape {body_pose.shape}"
        assert global_orient.shape == (num_frames, 3), f"global_orient shape {global_orient.shape}"
        assert transl.shape == (num_frames, 3), f"transl shape {transl.shape}"
        assert betas.shape == (10,), f"betas shape {betas.shape}"

        return {
            "fps": fps,
            "num_frames": int(num_frames),
            "body_pose": body_pose.astype(np.float32),
            "global_orient": global_orient.astype(np.float32),
            "transl": transl.astype(np.float32),
            "betas": betas.astype(np.float32),
        }

    # ─────────────────────────────────────────────────────────────────────
    # SMPL rest mesh + rig extraction (for Blender)
    # ─────────────────────────────────────────────────────────────────────

    def _export_smpl_rig_and_mesh(self, smpl_params: dict, out_dir: Path) -> None:
        """
        Load the SMPL body model and export the rest-pose rig and skinned
        mesh data that Blender needs. This avoids needing smplx/torch inside
        Blender's bundled Python.
        """
        import torch
        import smplx

        betas = torch.tensor(smpl_params["betas"]).unsqueeze(0)  # (1, 10)

        body_model = smplx.create(
            model_path=str(self.config.smpl_model_dir),
            model_type="smpl",
            gender=self.config.smpl_gender,
            num_betas=10,
            batch_size=1,
        )

        # Evaluate the model at rest pose with these betas to get the
        # shape-adjusted rest joint positions and vertex positions.
        with torch.no_grad():
            out = body_model(
                betas=betas,
                body_pose=torch.zeros(1, 69),
                global_orient=torch.zeros(1, 3),
                transl=torch.zeros(1, 3),
            )

        rest_joints = out.joints[0, :24].cpu().numpy()   # (24, 3) SMPL joints only
        rest_vertices = out.vertices[0].cpu().numpy()    # (6890, 3)

        # SMPL kinematic tree: parents[i] gives the parent joint index for i.
        # Loaded from the model's own `parents` buffer.
        parents = body_model.parents[:24].cpu().numpy().astype(np.int32)  # (24,)
        parents[0] = -1  # pelvis has no parent

        # Skinning weights and faces from the underlying SMPL model
        weights = body_model.lbs_weights.cpu().numpy()    # (6890, 24)
        faces = body_model.faces.astype(np.int32)         # (13776, 3)

        # Save rig metadata as JSON (small, human-readable)
        rig = {
            "joint_names": SMPL_JOINT_NAMES,
            "rest_joints": rest_joints.tolist(),
            "parents": parents.tolist(),
            "betas": smpl_params["betas"].tolist(),
        }
        (out_dir / "smpl_rig.json").write_text(json.dumps(rig, separators=(",", ":")))

        # Save mesh + weights as npz (binary, compact)
        np.savez_compressed(
            out_dir / "smpl_mesh.npz",
            v_template=rest_vertices.astype(np.float32),
            faces=faces,
            weights=weights.astype(np.float32),
        )

        logger.info(
            "Exported SMPL rig (%d joints) and mesh (%d verts, %d faces)",
            len(rest_joints), len(rest_vertices), len(faces),
        )

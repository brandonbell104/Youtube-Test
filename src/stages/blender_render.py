"""
Stage 7: Blender headless render — build SMPL avatar from tracking data,
apply SMPL pose + lip sync, render video.

Calls Blender as a subprocess with a Python script that:
  1. Builds an SMPL armature + skinned mesh from smpl_rig.json + smpl_mesh.npz
     (exported by the tracking stage).
  2. Applies per-frame SMPL pose parameters from tracking.json.
  3. Applies lip sync viseme data to blend shapes (no-op if the mesh has none).
  4. Sets up audio (voice.wav).
  5. Renders the final video.

Outputs:
  - output.mp4 — the final rendered video with new audio
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class BlenderRenderStage(Stage):
    name = "blender_render"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        # Gather all inputs
        tracking_dir = self.prev_stage_dir(job_id, "tracking")
        tracking_path = tracking_dir / "tracking.json"
        smpl_rig = tracking_dir / "smpl_rig.json"
        smpl_mesh = tracking_dir / "smpl_mesh.npz"
        lipsync_path = self.prev_stage_dir(job_id, "lipsync") / "lipsync.json"
        voice_path = self.prev_stage_dir(job_id, "voice") / "voice.wav"

        for path, label in [
            (tracking_path, "tracking data"),
            (smpl_rig, "SMPL rig"),
            (smpl_mesh, "SMPL mesh"),
            (lipsync_path, "lip sync data"),
            (voice_path, "voice audio"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")

        # Probe the narration length so the Blender script can time-stretch
        # the motion to match it (motion_sync == "stretch").
        audio_duration = self._get_audio_duration(voice_path)

        # Build the render config that the Blender script will read
        render_config = {
            "tracking_path": str(tracking_path),
            "lipsync_path": str(lipsync_path),
            "voice_path": str(voice_path),
            "output_path": str(out_dir / "output.mp4"),
            "resolution_x": self.config.render_resolution_x,
            "resolution_y": self.config.render_resolution_y,
            "fps": self.config.render_fps,
            "audio_duration": audio_duration,
            "motion_sync": self.config.motion_sync,
        }

        config_path = out_dir / "render_config.json"
        self.write_json(config_path, render_config)

        # Run Blender headless
        blender_script = (
            Path(__file__).resolve().parent.parent / "blender_scripts" / "render.py"
        )

        logger.info("Starting Blender render (headless)")
        cmd = [
            self.config.blender_bin,
            "--background",
            "--python", str(blender_script),
            "--", str(config_path),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for long renders
        )

        # Always write logs for debugging
        (out_dir / "blender_stdout.log").write_text(proc.stdout)
        (out_dir / "blender_stderr.log").write_text(proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Blender render failed (exit code {proc.returncode}). "
                f"Check {out_dir}/blender_stderr.log"
            )

        output_path = out_dir / "output.mp4"
        if not output_path.exists():
            raise FileNotFoundError("Blender did not produce output video")

        logger.info("Render complete: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
        return {"output_path": str(output_path)}

    @staticmethod
    def _get_audio_duration(path: Path) -> float:
        """Duration of an audio file in seconds via ffprobe (0.0 on failure)."""
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries",
                "format=duration", "-of", "csv=p=0", str(path),
            ],
            capture_output=True,
            text=True,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0


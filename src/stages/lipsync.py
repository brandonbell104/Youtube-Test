"""
Stage 6: Generate lip sync data from voice audio using Rhubarb Lip Sync.

Rhubarb analyzes the audio and produces phoneme/viseme timing data
that Blender uses to drive mouth blend shapes on the avatar.

Outputs:
  - lipsync.json — phoneme timing data (Rhubarb JSON format)
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)

# Rhubarb mouth shapes mapping to common blend shape names
VISEME_MAP = {
    "A": "viseme_aa",   # jaw open (as in "bat")
    "B": "viseme_PP",   # lips closed (as in "bat", "bump")
    "C": "viseme_I",    # tongue behind teeth (as in "sit")
    "D": "viseme_O",    # lips round (as in "hot")
    "E": "viseme_U",    # lips tight round (as in "boot")
    "F": "viseme_FF",   # lower lip on teeth (as in "fan")
    "G": "viseme_TH",   # tongue between teeth (as in "think")
    "H": "viseme_DD",   # tongue behind teeth (as in "dog")
    "X": "viseme_sil",  # silence / mouth closed
}


class LipSyncStage(Stage):
    name = "lipsync"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        voice_path = self.prev_stage_dir(job_id, "voice") / "voice.wav"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice audio not found: {voice_path}")

        # Also pass the transcript for better accuracy
        transcript_path = self.prev_stage_dir(job_id, "rewrite") / "rewritten_transcript.txt"
        dialog_file = None

        if transcript_path.exists():
            # Rhubarb can use a dialog file to improve recognition
            dialog_file = out_dir / "dialog.txt"
            dialog_file.write_text(
                transcript_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

        output_path = out_dir / "lipsync_raw.json"
        logger.info("Running Rhubarb Lip Sync on %s", voice_path)

        cmd = [
            self.config.rhubarb_bin,
            str(voice_path),
            "-f", "json",           # JSON output format
            "-o", str(output_path),
            "--recognizer", "phonetic",  # More accurate recognizer
        ]

        if dialog_file:
            cmd.extend(["-d", str(dialog_file)])

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0:
            raise RuntimeError(f"Rhubarb failed: {proc.stderr}")

        if not output_path.exists():
            raise FileNotFoundError("Rhubarb produced no output")

        # Parse and enhance the output
        raw_data = json.loads(output_path.read_text())
        mouth_cues = raw_data.get("mouthCues", [])

        # Create enhanced lip sync data with blend shape names
        enhanced_cues = []
        for cue in mouth_cues:
            enhanced_cues.append({
                "start": cue["start"],
                "end": cue["end"],
                "shape": cue["value"],
                "blend_shape": VISEME_MAP.get(cue["value"], "viseme_sil"),
            })

        lipsync_data = {
            "duration": raw_data.get("metadata", {}).get("duration", 0),
            "cue_count": len(enhanced_cues),
            "viseme_map": VISEME_MAP,
            "cues": enhanced_cues,
        }

        self.write_json(out_dir / "lipsync.json", lipsync_data)

        logger.info("Lip sync complete: %d mouth cues", len(enhanced_cues))
        return {"cue_count": len(enhanced_cues), "duration": lipsync_data["duration"]}

"""
Stage 4: Generate AI voice audio from rewritten transcript using Piper TTS.

Outputs:
  - voice.wav — the generated audio
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class VoiceStage(Stage):
    name = "voice"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        # Read the rewritten transcript (or user-edited version)
        rewrite_dir = self.prev_stage_dir(job_id, "rewrite")
        transcript_path = rewrite_dir / "rewritten_transcript.txt"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Rewritten transcript not found: {transcript_path}")

        text = transcript_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Rewritten transcript is empty")

        voice_path = out_dir / "voice.wav"
        model_dir = self.config.models_dir / "piper"
        model_dir.mkdir(parents=True, exist_ok=True)

        voice_name = self.config.piper_voice
        model_file = model_dir / f"{voice_name}.onnx"
        config_file = model_dir / f"{voice_name}.onnx.json"

        # Download voice model if not present
        if not model_file.exists():
            self._download_voice_model(voice_name, model_dir)

        logger.info("Generating voice with Piper (voice=%s, %d chars)", voice_name, len(text))

        # Run Piper TTS
        cmd = [
            self.config.piper_bin,
            "--model", str(model_file),
            "--config", str(config_file),
            "--output_file", str(voice_path),
            "--sentence_silence", "0.3",
        ]

        proc = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"Piper TTS failed: {proc.stderr}")

        if not voice_path.exists():
            raise FileNotFoundError("Piper did not produce output audio")

        # Get audio duration
        duration = self._get_audio_duration(voice_path)
        logger.info("Voice generated: %.1fs audio", duration)

        return {"voice_path": str(voice_path), "duration_seconds": duration}

    def _download_voice_model(self, voice_name: str, model_dir: Path) -> None:
        """Download a Piper voice model from the official repository."""
        logger.info("Downloading Piper voice model: %s", voice_name)

        # Piper models are hosted on Hugging Face
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

        # Parse voice name: en_US-lessac-high → en/en_US/lessac/high/
        parts = voice_name.split("-")
        lang_code = parts[0]  # en_US
        lang = lang_code.split("_")[0]  # en
        speaker = parts[1] if len(parts) > 1 else "default"
        quality = parts[2] if len(parts) > 2 else "medium"

        model_url = f"{base_url}/{lang}/{lang_code}/{speaker}/{quality}/{voice_name}.onnx"
        config_url = f"{model_url}.json"

        for url, filename in [
            (model_url, f"{voice_name}.onnx"),
            (config_url, f"{voice_name}.onnx.json"),
        ]:
            dest = model_dir / filename
            logger.info("Downloading %s", url)
            subprocess.run(
                ["wget", "-q", "-O", str(dest), url],
                check=True,
                timeout=300,
            )

    @staticmethod
    def _get_audio_duration(path: Path) -> float:
        """Get duration of a wav file using ffprobe."""
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

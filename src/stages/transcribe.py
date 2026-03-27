"""
Stage 2: Transcribe video audio using faster-whisper (large-v3).

Outputs:
  - transcript.json — word-level timestamps
  - transcript.txt  — plain text (for easy editing)
"""

import logging
from pathlib import Path
from typing import Any

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class TranscribeStage(Stage):
    name = "transcribe"

    def __init__(self, config: Config):
        super().__init__(config)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model: %s (device=%s, compute=%s)",
                self.config.whisper_model,
                self.config.whisper_device,
                self.config.whisper_compute_type,
            )
            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
        return self._model

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        video_path = self.prev_stage_dir(job_id, "download") / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        model = self._get_model()
        logger.info("Transcribing %s", video_path)

        segments, info = model.transcribe(
            str(video_path),
            beam_size=5,
            word_timestamps=True,
            language="en",
            condition_on_previous_text=True,
            vad_filter=True,
        )

        # Collect segments with word-level timestamps
        transcript_data = {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "segments": [],
        }

        full_text_parts = []

        for segment in segments:
            seg_data = {
                "id": segment.id,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": [],
            }
            if segment.words:
                for w in segment.words:
                    seg_data["words"].append({
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    })
            transcript_data["segments"].append(seg_data)
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)

        # Write structured transcript
        self.write_json(out_dir / "transcript.json", transcript_data)

        # Write plain text for easy editing
        (out_dir / "transcript.txt").write_text(full_text, encoding="utf-8")

        logger.info(
            "Transcription complete: %d segments, %d words",
            len(transcript_data["segments"]),
            sum(len(s["words"]) for s in transcript_data["segments"]),
        )

        # Release GPU memory
        self._model = None

        return {
            "segment_count": len(transcript_data["segments"]),
            "word_count": len(full_text.split()),
            "duration": transcript_data["duration"],
        }

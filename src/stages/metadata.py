"""
Stage 8: Generate video metadata (title, description, tags, thumbnail).

Uses the local LLM (Ollama) to generate YouTube metadata based on the
transcript. Extracts a representative frame for thumbnail generation.

Outputs:
  - metadata.json    — title, description, tags
  - thumbnail.png    — extracted/generated thumbnail
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import cv2
import requests

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)

METADATA_PROMPT = """Based on the following video transcript, generate YouTube metadata.

Transcript:
{transcript}

Generate the following in JSON format (and nothing else):
{{
    "title": "A catchy, SEO-friendly title (under 70 characters)",
    "description": "A compelling description (150-300 words) with relevant keywords. Include a brief summary of what the video covers.",
    "tags": ["tag1", "tag2", "tag3", ...] (10-15 relevant tags for SEO)
}}

The video is a yoga instruction video. Make the metadata engaging and searchable.
Output valid JSON only."""


class MetadataStage(Stage):
    name = "metadata"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        # Read the rewritten transcript
        transcript_path = (
            self.prev_stage_dir(job_id, "rewrite") / "rewritten_transcript.txt"
        )
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        transcript = transcript_path.read_text(encoding="utf-8").strip()

        # Generate metadata with LLM
        logger.info("Generating metadata with %s", self.config.llm_model)
        metadata = self._generate_metadata(transcript)

        # Extract thumbnail from rendered video
        video_path = self.prev_stage_dir(job_id, "blender_render") / "output.mp4"
        thumbnail_path = out_dir / "thumbnail.png"

        if video_path.exists():
            self._extract_thumbnail(video_path, thumbnail_path)
        else:
            # Fallback: use original video
            orig_video = self.prev_stage_dir(job_id, "download") / "video.mp4"
            if orig_video.exists():
                self._extract_thumbnail(orig_video, thumbnail_path)

        # Write metadata
        metadata["thumbnail_path"] = str(thumbnail_path) if thumbnail_path.exists() else None
        self.write_json(out_dir / "metadata.json", metadata)

        logger.info("Metadata generated: '%s'", metadata.get("title", ""))
        return metadata

    def _generate_metadata(self, transcript: str) -> dict[str, Any]:
        # Truncate transcript if too long for context
        if len(transcript) > 8000:
            transcript = transcript[:8000] + "..."

        prompt = METADATA_PROMPT.format(transcript=transcript)

        url = f"{self.config.ollama_host}/api/generate"
        payload = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.7,
                "num_predict": 2048,
            },
        }

        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        response_text = resp.json()["response"].strip()

        try:
            metadata = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                metadata = json.loads(response_text[start:end])
            else:
                metadata = {
                    "title": "Yoga Flow Session",
                    "description": transcript[:300],
                    "tags": ["yoga", "fitness", "wellness"],
                }

        return metadata

    @staticmethod
    def _extract_thumbnail(video_path: Path, output_path: Path) -> None:
        """Extract a visually appealing frame from the video as thumbnail."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample frame at ~30% into the video (usually a good pose)
        target_frame = int(total_frames * 0.3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        cap.release()

        if ret:
            # Center-crop to 16:9 first so the thumbnail isn't stretched,
            # then resize to YouTube dimensions (1280x720).
            h, w = frame.shape[:2]
            target_ratio = 16 / 9
            if w / h > target_ratio:
                new_w = int(h * target_ratio)
                x0 = (w - new_w) // 2
                frame = frame[:, x0:x0 + new_w]
            else:
                new_h = int(w / target_ratio)
                y0 = (h - new_h) // 2
                frame = frame[y0:y0 + new_h, :]
            thumb = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(output_path), thumb)

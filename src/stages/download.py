"""
Stage 1: Download video from YouTube using yt-dlp.

Outputs:
  - video.mp4          — best quality video+audio
  - source_meta.json   — title, description, uploader, duration, etc.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yt_dlp

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class DownloadStage(Stage):
    name = "download"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        job = self._get_job_url(job_id)
        url = job

        video_path = out_dir / "video.mp4"
        meta_path = out_dir / "source_meta.json"

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(video_path.with_suffix("")),  # yt-dlp adds extension
            "merge_output_format": "mp4",
            "writeinfojson": False,
            "quiet": True,
            "no_warnings": True,
        }

        logger.info("Downloading %s", url)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Handle yt-dlp possibly appending extension
        if not video_path.exists():
            # Try the common pattern where yt-dlp names it without double extension
            candidate = out_dir / "video.mp4"
            if not candidate.exists():
                # Look for any mp4 in the directory
                mp4s = list(out_dir.glob("*.mp4"))
                if mp4s:
                    mp4s[0].rename(video_path)
                else:
                    raise FileNotFoundError(f"Download produced no mp4 in {out_dir}")

        # Extract useful metadata
        source_meta = {
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "uploader": info.get("uploader", ""),
            "duration": info.get("duration", 0),
            "view_count": info.get("view_count", 0),
            "upload_date": info.get("upload_date", ""),
            "url": url,
            "original_id": info.get("id", ""),
        }
        self.write_json(meta_path, source_meta)

        logger.info(
            "Downloaded: %s (%.1f MB, %ds)",
            source_meta["title"],
            video_path.stat().st_size / 1e6,
            source_meta["duration"],
        )

        return {
            "video_path": str(video_path),
            "title": source_meta["title"],
            "duration": source_meta["duration"],
        }

    def _get_job_url(self, job_id: str) -> str:
        """Retrieve the URL for this job from the database."""
        from src.pipeline.job_queue import JobQueue

        queue = JobQueue(self.config)
        job = queue.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        return job["url"]

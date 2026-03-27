"""
Stage 9: Upload video to YouTube using the YouTube Data API v3.

This stage is skipped if YouTube credentials are not configured.
The user can always upload manually and use this stage later.

Outputs:
  - upload_result.json — YouTube video ID, URL, status
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class UploadStage(Stage):
    name = "upload"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        if not self.config.youtube_client_id:
            logger.info("YouTube API not configured — skipping upload")
            return {"skipped": True, "reason": "YouTube API not configured"}

        # Load metadata
        metadata_path = self.prev_stage_dir(job_id, "metadata") / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = self.read_json(metadata_path)

        # Find the rendered video
        video_path = self.prev_stage_dir(job_id, "blender_render") / "output.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        thumbnail_path = self.prev_stage_dir(job_id, "metadata") / "thumbnail.png"

        logger.info("Uploading to YouTube: '%s'", metadata.get("title", ""))

        # Perform the upload
        result = self._upload_video(
            video_path=video_path,
            title=metadata.get("title", "Yoga Video"),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", []),
            thumbnail_path=thumbnail_path if thumbnail_path.exists() else None,
        )

        self.write_json(out_dir / "upload_result.json", result)

        logger.info("Upload complete: https://youtu.be/%s", result.get("video_id", ""))
        return result

    def _upload_video(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: list[str],
        thumbnail_path: Path | None = None,
    ) -> dict[str, Any]:
        """Upload video using Google API client."""
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

        # Load or create credentials
        creds = None
        token_path = self.config.youtube_token_path

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_config(
                    {
                        "installed": {
                            "client_id": self.config.youtube_client_id,
                            "client_secret": self.config.youtube_client_secret,
                            "redirect_uris": ["http://localhost"],
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                        }
                    },
                    SCOPES,
                )
                creds = flow.run_local_server(port=0)

            # Save credentials
            token_path.write_text(creds.to_json())

        youtube = build("youtube", "v3", credentials=creds)

        # Upload video (as unlisted by default — user publishes via UI)
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": "26",  # Howto & Style
            },
            "status": {
                "privacyStatus": "unlisted",  # Safe default — user publishes manually
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)

        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
        )

        response = request.execute()
        video_id = response["id"]

        # Set thumbnail if available
        if thumbnail_path:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/png"),
            ).execute()

        return {
            "video_id": video_id,
            "url": f"https://youtu.be/{video_id}",
            "status": response["status"]["uploadStatus"],
            "privacy": response["status"]["privacyStatus"],
        }

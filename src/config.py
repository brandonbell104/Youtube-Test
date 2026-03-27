"""
Central configuration for the YouTube Automation Pipeline.
All paths, model names, and tunables live here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    workspace_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    avatars_dir: Path = field(default=None)
    db_path: Path = field(default=None)

    # ── Whisper ──────────────────────────────────────────────────────────
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "large-v3"))
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # ── Ollama / LLM ────────────────────────────────────────────────────
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama3:8b"))

    # ── Piper TTS ────────────────────────────────────────────────────────
    piper_voice: str = field(default_factory=lambda: _env("PIPER_VOICE", "en_US-lessac-high"))
    piper_bin: str = field(default_factory=lambda: _env("PIPER_BIN", "piper"))

    # ── Rhubarb Lip Sync ────────────────────────────────────────────────
    rhubarb_bin: str = field(default_factory=lambda: _env("RHUBARB_BIN", "rhubarb"))

    # ── MediaPipe ────────────────────────────────────────────────────────
    tracking_model_complexity: int = 2  # 0, 1, or 2 (2 = most accurate)
    tracking_min_detection_confidence: float = 0.7
    tracking_min_tracking_confidence: float = 0.7

    # ── Blender ──────────────────────────────────────────────────────────
    blender_bin: str = field(default_factory=lambda: _env("BLENDER_BIN", "blender"))
    render_resolution_x: int = field(
        default_factory=lambda: int(_env("RENDER_RESOLUTION_X", "1920"))
    )
    render_resolution_y: int = field(
        default_factory=lambda: int(_env("RENDER_RESOLUTION_Y", "1080"))
    )
    render_fps: int = field(default_factory=lambda: int(_env("RENDER_FPS", "30")))
    default_avatar: str = "default_avatar.glb"

    # ── YouTube API ──────────────────────────────────────────────────────
    youtube_client_id: str = field(default_factory=lambda: _env("YOUTUBE_CLIENT_ID"))
    youtube_client_secret: str = field(default_factory=lambda: _env("YOUTUBE_CLIENT_SECRET"))
    youtube_token_path: Path = field(default=None)

    # ── Web UI ───────────────────────────────────────────────────────────
    web_host: str = "0.0.0.0"
    web_port: int = field(default_factory=lambda: int(_env("WEB_PORT", "7860")))

    def __post_init__(self):
        if self.workspace_dir is None:
            self.workspace_dir = self.base_dir / "workspace"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.avatars_dir is None:
            self.avatars_dir = self.base_dir / "avatars"
        if self.db_path is None:
            self.db_path = self.workspace_dir / "jobs.db"
        if self.youtube_token_path is None:
            self.youtube_token_path = self.base_dir / "youtube_token.json"

        # Ensure directories exist
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.avatars_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Return the working directory for a specific job."""
        p = self.workspace_dir / job_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stage_dir(self, job_id: str, stage_name: str) -> Path:
        """Return the output directory for a specific stage of a job."""
        p = self.job_dir(job_id) / stage_name
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton
config = Config()

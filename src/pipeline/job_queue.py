"""
SQLite-backed job queue with per-stage tracking.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import Config


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Ordered list of all pipeline stages
STAGE_ORDER = [
    "download",
    "transcribe",
    "rewrite",
    "voice",
    "tracking",
    "lipsync",
    "blender_render",
    "metadata",
    "upload",
]


class JobQueue:
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    status TEXT DEFAULT 'queued',
                    priority INTEGER DEFAULT 0,
                    current_stage TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_stages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    stage_name TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}',
                    started_at TEXT,
                    completed_at TEXT,
                    UNIQUE(job_id, stage_name)
                );
            """)

    # ── Job CRUD ─────────────────────────────────────────────────────────

    def add_job(self, url: str, priority: int = 0) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, url, status, priority, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, url, JobStatus.QUEUED, priority, now, now),
            )
            # Pre-create all stage rows
            for stage in STAGE_ORDER:
                conn.execute(
                    "INSERT INTO job_stages (job_id, stage_name, status) VALUES (?, ?, ?)",
                    (job_id, stage, StageStatus.PENDING),
                )
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            job = dict(row)
            stages = conn.execute(
                "SELECT * FROM job_stages WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
            job["stages"] = [dict(s) for s in stages]
            return job

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY priority DESC, created_at ASC"
            ).fetchall()
            jobs = []
            for row in rows:
                job = dict(row)
                stages = conn.execute(
                    "SELECT * FROM job_stages WHERE job_id = ? ORDER BY id",
                    (job["id"],),
                ).fetchall()
                job["stages"] = [dict(s) for s in stages]
                jobs.append(job)
            return jobs

    def delete_job(self, job_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    # ── Job status updates ───────────────────────────────────────────────

    def update_job_status(self, job_id: str, status: JobStatus, **kwargs) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sets = ["status = ?", "updated_at = ?"]
        vals: list[Any] = [status, now]

        for key in ("current_stage", "error_message", "title"):
            if key in kwargs:
                sets.append(f"{key} = ?")
                vals.append(kwargs[key])

        vals.append(job_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals
            )

    def update_stage_status(
        self,
        job_id: str,
        stage_name: str,
        status: StageStatus,
        error_message: str = "",
        metadata: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

        with self._conn() as conn:
            if status == StageStatus.RUNNING:
                conn.execute(
                    "UPDATE job_stages SET status=?, started_at=?, error_message=? "
                    "WHERE job_id=? AND stage_name=?",
                    (status, now, error_message, job_id, stage_name),
                )
            elif status in (StageStatus.COMPLETED, StageStatus.FAILED):
                conn.execute(
                    "UPDATE job_stages SET status=?, completed_at=?, error_message=?, metadata=? "
                    "WHERE job_id=? AND stage_name=?",
                    (status, now, error_message, meta_json, job_id, stage_name),
                )
            else:
                conn.execute(
                    "UPDATE job_stages SET status=?, error_message=? "
                    "WHERE job_id=? AND stage_name=?",
                    (status, error_message, job_id, stage_name),
                )

    # ── Queue operations ─────────────────────────────────────────────────

    def next_job(self) -> dict[str, Any] | None:
        """Get the next queued job (highest priority, oldest first)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY priority DESC, created_at ASC LIMIT 1",
                (JobStatus.QUEUED,),
            ).fetchone()
            if row is None:
                return None
            return self.get_job(row["id"])

    def get_next_pending_stage(self, job_id: str) -> str | None:
        """Get the name of the next pending stage for a job."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT stage_name FROM job_stages WHERE job_id = ? AND status = ? ORDER BY id LIMIT 1",
                (job_id, StageStatus.PENDING),
            ).fetchone()
            return row["stage_name"] if row else None

    def reset_stage_from(self, job_id: str, stage_name: str) -> None:
        """Reset a stage and all subsequent stages to pending (for re-running after edit)."""
        try:
            start_idx = STAGE_ORDER.index(stage_name)
        except ValueError:
            return

        stages_to_reset = STAGE_ORDER[start_idx:]
        with self._conn() as conn:
            for s in stages_to_reset:
                conn.execute(
                    "UPDATE job_stages SET status=?, error_message='', "
                    "started_at=NULL, completed_at=NULL, metadata='{}' "
                    "WHERE job_id=? AND stage_name=?",
                    (StageStatus.PENDING, job_id, s),
                )

    def pause_job(self, job_id: str) -> None:
        self.update_job_status(job_id, JobStatus.PAUSED)

    def resume_job(self, job_id: str) -> None:
        self.update_job_status(job_id, JobStatus.QUEUED)

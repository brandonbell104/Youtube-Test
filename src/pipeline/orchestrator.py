"""
Pipeline orchestrator — runs jobs through stages sequentially,
respects pause requests, and processes the queue continuously.
"""

import logging
import threading
import time
import traceback
from typing import Any

from src.config import Config
from src.pipeline.job_queue import (
    JobQueue,
    JobStatus,
    StageStatus,
    STAGE_ORDER,
)
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Pulls jobs from the queue and runs them through pipeline stages.
    Runs in a background thread so the web UI stays responsive.
    """

    def __init__(self, config: Config, stages: dict[str, Stage]):
        self.config = config
        self.queue = JobQueue(config)
        self.stages = stages  # name → Stage instance
        self._running = False
        self._thread: threading.Thread | None = None
        self._pause_requested: dict[str, bool] = {}  # job_id → True if pause requested

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the orchestrator loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Orchestrator started")

    def stop(self) -> None:
        """Signal the orchestrator to stop after the current stage finishes."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Orchestrator stopped")

    # ── Public controls ──────────────────────────────────────────────────

    def add_job(self, url: str, priority: int = 0) -> str:
        return self.queue.add_job(url, priority)

    def pause_job(self, job_id: str) -> None:
        """Request a pause — takes effect between stages."""
        self._pause_requested[job_id] = True
        self.queue.pause_job(job_id)
        logger.info("Pause requested for job %s", job_id)

    def resume_job(self, job_id: str) -> None:
        """Resume a paused job."""
        self._pause_requested.pop(job_id, None)
        self.queue.resume_job(job_id)
        logger.info("Resumed job %s", job_id)

    def restart_stage(self, job_id: str, stage_name: str) -> None:
        """Reset this stage and all following stages, then re-queue the job."""
        self.queue.reset_stage_from(job_id, stage_name)
        self.queue.update_job_status(job_id, JobStatus.QUEUED)
        self._pause_requested.pop(job_id, None)
        logger.info("Restarting job %s from stage %s", job_id, stage_name)

    def delete_job(self, job_id: str) -> None:
        self._pause_requested.pop(job_id, None)
        self.queue.delete_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.queue.get_job(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.queue.list_jobs()

    # ── Main loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            job = self.queue.next_job()
            if job is None:
                time.sleep(2)  # No jobs queued — poll
                continue
            self._run_job(job)

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        self.queue.update_job_status(job_id, JobStatus.RUNNING)
        logger.info("Processing job %s (%s)", job_id, job["url"])

        for stage_name in STAGE_ORDER:
            # Check for pause
            if self._pause_requested.get(job_id):
                logger.info("Job %s paused before stage %s", job_id, stage_name)
                self.queue.update_job_status(job_id, JobStatus.PAUSED)
                return

            if not self._running:
                self.queue.update_job_status(job_id, JobStatus.PAUSED)
                return

            # Skip already-completed stages
            stage_info = self._get_stage_info(job, stage_name)
            if stage_info and stage_info["status"] == StageStatus.COMPLETED:
                continue

            # Skip upload if not configured
            if stage_name == "upload" and not self.config.youtube_client_id:
                self.queue.update_stage_status(job_id, stage_name, StageStatus.SKIPPED)
                continue

            # Run the stage
            stage = self.stages.get(stage_name)
            if stage is None:
                logger.warning("No implementation for stage %s, skipping", stage_name)
                self.queue.update_stage_status(job_id, stage_name, StageStatus.SKIPPED)
                continue

            self.queue.update_job_status(job_id, JobStatus.RUNNING, current_stage=stage_name)
            self.queue.update_stage_status(job_id, stage_name, StageStatus.RUNNING)

            try:
                result = stage.execute(job_id)
                self.queue.update_stage_status(
                    job_id, stage_name, StageStatus.COMPLETED, metadata=result
                )
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Stage %s failed for job %s: %s\n%s", stage_name, job_id, e, tb)
                self.queue.update_stage_status(
                    job_id, stage_name, StageStatus.FAILED, error_message=str(e)
                )
                self.queue.update_job_status(
                    job_id, JobStatus.FAILED, error_message=f"Stage {stage_name}: {e}"
                )
                return

        # All stages complete
        self.queue.update_job_status(job_id, JobStatus.COMPLETED)
        logger.info("Job %s completed successfully", job_id)

    @staticmethod
    def _get_stage_info(job: dict, stage_name: str) -> dict | None:
        for s in job.get("stages", []):
            if s["stage_name"] == stage_name:
                return s
        return None

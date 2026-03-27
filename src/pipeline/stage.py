"""
Base class for all pipeline stages.

Every stage follows the same contract:
  1. Receives a job_id and the shared Config
  2. Reads its inputs from previous stages' output directories
  3. Writes its outputs to its own stage directory
  4. Reports status via callbacks
"""

import abc
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import Config

logger = logging.getLogger(__name__)


class Stage(abc.ABC):
    """Abstract base for a pipeline stage."""

    name: str = "base"  # Override in subclass

    def __init__(self, config: Config):
        self.config = config

    # ── Public API ───────────────────────────────────────────────────────

    def execute(self, job_id: str) -> dict[str, Any]:
        """
        Run this stage for the given job.
        Returns a dict of metadata about the run (timings, output paths, etc.).
        """
        out_dir = self.config.stage_dir(job_id, self.name)
        logger.info("Stage [%s] starting for job %s → %s", self.name, job_id, out_dir)

        start = time.time()
        result = self.run(job_id, out_dir)
        elapsed = time.time() - start

        meta = {
            "stage": self.name,
            "job_id": job_id,
            "elapsed_seconds": round(elapsed, 2),
            "output_dir": str(out_dir),
            **(result or {}),
        }

        # Persist stage metadata
        meta_path = out_dir / "stage_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info("Stage [%s] completed in %.1fs", self.name, elapsed)
        return meta

    # ── Subclass contract ────────────────────────────────────────────────

    @abc.abstractmethod
    def run(self, job_id: str, out_dir: Path) -> dict[str, Any] | None:
        """
        Implement the actual work of the stage.
        Write all outputs into out_dir.
        Optionally return a dict of extra metadata.
        """
        ...

    # ── Helpers ──────────────────────────────────────────────────────────

    def prev_stage_dir(self, job_id: str, stage_name: str) -> Path:
        """Get the output directory of a previous stage."""
        return self.config.stage_dir(job_id, stage_name)

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())

    def write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

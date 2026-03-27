"""
Stage 3: Rewrite transcript using a local LLM via Ollama.

Reads transcript.txt (which the user may have manually edited),
sends it to the LLM for rewording, writes the result.

Outputs:
  - rewritten_transcript.txt  — the reworded text
  - rewrite_log.json           — prompt and model info
"""

import logging
from pathlib import Path
from typing import Any

import requests

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional script rewriter. Your task is to reword
the following transcript so that it conveys the same information and instructions
but uses different phrasing, sentence structure, and word choices.

Rules:
- Preserve the MEANING and ORDER of all instructions exactly.
- Change wording, synonyms, sentence structure, and phrasing.
- Keep the same tone (instructional, calm, guiding).
- Do NOT add new content, opinions, or remove any instructions.
- Do NOT add introductions, conclusions, or meta-commentary.
- Output ONLY the rewritten transcript text, nothing else.
- Maintain similar paragraph breaks as the original.
- If the original references specific pose names (e.g., "downward dog",
  "warrior two"), keep those exact names — only reword the surrounding text."""


class RewriteStage(Stage):
    name = "rewrite"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        transcript_path = self.prev_stage_dir(job_id, "transcribe") / "transcript.txt"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        original_text = transcript_path.read_text(encoding="utf-8").strip()
        if not original_text:
            raise ValueError("Transcript is empty")

        logger.info(
            "Rewriting transcript (%d chars) with %s",
            len(original_text),
            self.config.llm_model,
        )

        # Split into chunks if very long (>3000 words) to stay within context
        chunks = self._split_into_chunks(original_text, max_words=2500)
        rewritten_parts = []

        for i, chunk in enumerate(chunks):
            logger.info("Processing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            rewritten = self._call_ollama(chunk)
            rewritten_parts.append(rewritten)

        rewritten_text = "\n\n".join(rewritten_parts)

        # Write outputs
        (out_dir / "rewritten_transcript.txt").write_text(rewritten_text, encoding="utf-8")

        log_data = {
            "model": self.config.llm_model,
            "original_length": len(original_text),
            "rewritten_length": len(rewritten_text),
            "chunks": len(chunks),
            "system_prompt": SYSTEM_PROMPT,
        }
        self.write_json(out_dir / "rewrite_log.json", log_data)

        logger.info(
            "Rewrite complete: %d → %d chars", len(original_text), len(rewritten_text)
        )
        return {"original_chars": len(original_text), "rewritten_chars": len(rewritten_text)}

    def _call_ollama(self, text: str) -> str:
        url = f"{self.config.ollama_host}/api/generate"
        payload = {
            "model": self.config.llm_model,
            "prompt": f"Rewrite the following transcript:\n\n{text}",
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 8192,
            },
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["response"].strip()

    @staticmethod
    def _split_into_chunks(text: str, max_words: int = 2500) -> list[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        sentences = text.replace(".", ".\n").split("\n")
        current_chunk: list[str] = []
        current_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            s_words = len(sentence.split())
            if current_count + s_words > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0
            current_chunk.append(sentence)
            current_count += s_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

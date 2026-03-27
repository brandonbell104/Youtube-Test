"""
Gradio Web UI for the YouTube Automation Pipeline.

Tabs:
  1. Queue     — paste URLs, view/manage the job queue
  2. Pipeline  — per-job stage progress, pause/resume/restart
  3. Editor    — view/edit transcripts, re-run from edit point
  4. Review    — preview final video, edit metadata, publish
  5. Settings  — model, avatar, voice, output preferences
"""

import json
import logging
import sys
from pathlib import Path

import gradio as gr

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import config
from src.pipeline.orchestrator import Orchestrator
from src.pipeline.job_queue import STAGE_ORDER, JobStatus, StageStatus
from src.stages import (
    DownloadStage,
    TranscribeStage,
    RewriteStage,
    VoiceStage,
    TrackingStage,
    LipSyncStage,
    BlenderRenderStage,
    MetadataStage,
    UploadStage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Initialize pipeline ─────────────────────────────────────────────────

stages = {
    "download": DownloadStage(config),
    "transcribe": TranscribeStage(config),
    "rewrite": RewriteStage(config),
    "voice": VoiceStage(config),
    "tracking": TrackingStage(config),
    "lipsync": LipSyncStage(config),
    "blender_render": BlenderRenderStage(config),
    "metadata": MetadataStage(config),
    "upload": UploadStage(config),
}

orchestrator = Orchestrator(config, stages)
orchestrator.start()


# ── Helper functions ─────────────────────────────────────────────────────

def format_stage_status(stage: dict) -> str:
    status = stage["status"]
    icons = {
        "pending": "⏳",
        "running": "🔄",
        "paused": "⏸️",
        "completed": "✅",
        "failed": "❌",
        "skipped": "⏭️",
    }
    icon = icons.get(status, "❓")
    name = stage["stage_name"].replace("_", " ").title()
    line = f"{icon} {name}"
    if stage.get("error_message"):
        line += f" — {stage['error_message']}"
    return line


def get_queue_display() -> str:
    jobs = orchestrator.list_jobs()
    if not jobs:
        return "No jobs in queue. Paste a YouTube URL above to get started."

    lines = []
    for i, job in enumerate(jobs, 1):
        status_icons = {
            "queued": "🟡", "running": "🟢", "paused": "⏸️",
            "completed": "✅", "failed": "❌", "cancelled": "🚫",
        }
        icon = status_icons.get(job["status"], "❓")
        title = job.get("title") or job["url"]
        stage = job.get("current_stage", "")
        stage_display = f" [{stage.replace('_', ' ').title()}]" if stage else ""
        lines.append(f"{i}. {icon} **{title}**{stage_display}\n   `{job['id']}` — {job['status']}")

    return "\n\n".join(lines)


def get_job_choices() -> list[str]:
    jobs = orchestrator.list_jobs()
    return [f"{j['id']} — {j.get('title') or j['url']}" for j in jobs]


# ── Queue Tab ────────────────────────────────────────────────────────────

def add_to_queue(urls_text: str) -> tuple[str, str]:
    urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
    if not urls:
        return "No URLs provided.", get_queue_display()

    added = []
    for url in urls:
        job_id = orchestrator.add_job(url)
        added.append(f"Added: {url} → {job_id}")

    return "\n".join(added), get_queue_display()


def refresh_queue() -> str:
    return get_queue_display()


def remove_job(job_selector: str) -> tuple[str, str]:
    if not job_selector:
        return "No job selected.", get_queue_display()
    job_id = job_selector.split(" — ")[0].strip()
    orchestrator.delete_job(job_id)
    return f"Removed job {job_id}", get_queue_display()


# ── Pipeline Tab ─────────────────────────────────────────────────────────

def get_pipeline_status(job_selector: str) -> str:
    if not job_selector:
        return "Select a job to view its pipeline status."
    job_id = job_selector.split(" — ")[0].strip()
    job = orchestrator.get_job(job_id)
    if not job:
        return "Job not found."

    lines = [f"## Job: {job.get('title') or job['url']}", f"**Status:** {job['status']}", ""]
    lines.append("### Stages:")
    for stage in job.get("stages", []):
        lines.append(format_stage_status(stage))

    if job.get("error_message"):
        lines.append(f"\n**Error:** {job['error_message']}")

    return "\n".join(lines)


def pause_job_action(job_selector: str) -> str:
    if not job_selector:
        return "No job selected."
    job_id = job_selector.split(" — ")[0].strip()
    orchestrator.pause_job(job_id)
    return get_pipeline_status(job_selector)


def resume_job_action(job_selector: str) -> str:
    if not job_selector:
        return "No job selected."
    job_id = job_selector.split(" — ")[0].strip()
    orchestrator.resume_job(job_id)
    return get_pipeline_status(job_selector)


def restart_from_stage(job_selector: str, stage_name: str) -> str:
    if not job_selector or not stage_name:
        return "Select a job and stage."
    job_id = job_selector.split(" — ")[0].strip()
    orchestrator.restart_stage(job_id, stage_name)
    return get_pipeline_status(job_selector)


# ── Editor Tab ───────────────────────────────────────────────────────────

def load_transcript(job_selector: str, transcript_type: str) -> str:
    if not job_selector:
        return "Select a job first."
    job_id = job_selector.split(" — ")[0].strip()

    if transcript_type == "Original Transcript":
        path = config.stage_dir(job_id, "transcribe") / "transcript.txt"
    else:
        path = config.stage_dir(job_id, "rewrite") / "rewritten_transcript.txt"

    if not path.exists():
        return f"File not found: {path.name}. Run the pipeline first."
    return path.read_text(encoding="utf-8")


def save_transcript(job_selector: str, transcript_type: str, text: str) -> str:
    if not job_selector:
        return "Select a job first."
    job_id = job_selector.split(" — ")[0].strip()

    if transcript_type == "Original Transcript":
        path = config.stage_dir(job_id, "transcribe") / "transcript.txt"
        next_stage = "rewrite"
    else:
        path = config.stage_dir(job_id, "rewrite") / "rewritten_transcript.txt"
        next_stage = "voice"

    path.write_text(text, encoding="utf-8")
    return f"Saved! Use 'Restart from stage' with '{next_stage}' to re-run the pipeline from here."


def save_and_continue(job_selector: str, transcript_type: str, text: str) -> str:
    if not job_selector:
        return "Select a job first."
    job_id = job_selector.split(" — ")[0].strip()

    if transcript_type == "Original Transcript":
        path = config.stage_dir(job_id, "transcribe") / "transcript.txt"
        next_stage = "rewrite"
    else:
        path = config.stage_dir(job_id, "rewrite") / "rewritten_transcript.txt"
        next_stage = "voice"

    path.write_text(text, encoding="utf-8")
    orchestrator.restart_stage(job_id, next_stage)
    return f"Saved and restarting from '{next_stage}' stage."


# ── Review Tab ───────────────────────────────────────────────────────────

def load_review(job_selector: str) -> tuple[str | None, str, str, str, str | None]:
    if not job_selector:
        return None, "", "", "", None

    job_id = job_selector.split(" — ")[0].strip()

    # Video
    video_path = config.stage_dir(job_id, "blender_render") / "output.mp4"
    video = str(video_path) if video_path.exists() else None

    # Metadata
    meta_path = config.stage_dir(job_id, "metadata") / "metadata.json"
    title, description, tags = "", "", ""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        title = meta.get("title", "")
        description = meta.get("description", "")
        tags = ", ".join(meta.get("tags", []))

    # Thumbnail
    thumb_path = config.stage_dir(job_id, "metadata") / "thumbnail.png"
    thumbnail = str(thumb_path) if thumb_path.exists() else None

    return video, title, description, tags, thumbnail


def save_metadata(job_selector: str, title: str, description: str, tags: str) -> str:
    if not job_selector:
        return "Select a job first."
    job_id = job_selector.split(" — ")[0].strip()

    meta_path = config.stage_dir(job_id, "metadata") / "metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    meta["title"] = title
    meta["description"] = description
    meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    meta_path.write_text(json.dumps(meta, indent=2))
    return "Metadata saved."


def publish_video(job_selector: str) -> str:
    if not job_selector:
        return "Select a job first."
    if not config.youtube_client_id:
        return "YouTube API not configured. Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET."

    job_id = job_selector.split(" — ")[0].strip()
    orchestrator.restart_stage(job_id, "upload")
    return "Upload stage queued. The video will be uploaded as 'unlisted' — change visibility on YouTube."


# ── Build UI ─────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="YouTube Automation Pipeline",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# YouTube Automation Pipeline")
        gr.Markdown("Download → Transcribe → Rewrite → Voice → Track → Lip Sync → Render → Publish")

        # ── Queue Tab ──
        with gr.Tab("Queue"):
            with gr.Row():
                with gr.Column(scale=1):
                    url_input = gr.Textbox(
                        label="YouTube URLs (one per line)",
                        placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
                        lines=5,
                    )
                    with gr.Row():
                        add_btn = gr.Button("Add to Queue", variant="primary")
                        refresh_btn = gr.Button("Refresh")
                with gr.Column(scale=2):
                    queue_display = gr.Markdown(value=get_queue_display())

            add_status = gr.Textbox(label="Status", interactive=False)
            add_btn.click(add_to_queue, inputs=[url_input], outputs=[add_status, queue_display])
            refresh_btn.click(refresh_queue, outputs=[queue_display])

        # ── Pipeline Tab ──
        with gr.Tab("Pipeline"):
            with gr.Row():
                job_select_pipeline = gr.Dropdown(
                    label="Select Job",
                    choices=get_job_choices(),
                    interactive=True,
                )
                refresh_jobs_btn = gr.Button("Refresh Jobs")

            pipeline_display = gr.Markdown("Select a job to view pipeline status.")

            with gr.Row():
                pause_btn = gr.Button("Pause", variant="secondary")
                resume_btn = gr.Button("Resume", variant="primary")

            with gr.Row():
                stage_select = gr.Dropdown(
                    label="Restart from Stage",
                    choices=STAGE_ORDER,
                    interactive=True,
                )
                restart_btn = gr.Button("Restart from Stage", variant="stop")

            job_select_pipeline.change(
                get_pipeline_status, inputs=[job_select_pipeline], outputs=[pipeline_display]
            )
            refresh_jobs_btn.click(
                lambda: gr.update(choices=get_job_choices()),
                outputs=[job_select_pipeline],
            )
            pause_btn.click(
                pause_job_action, inputs=[job_select_pipeline], outputs=[pipeline_display]
            )
            resume_btn.click(
                resume_job_action, inputs=[job_select_pipeline], outputs=[pipeline_display]
            )
            restart_btn.click(
                restart_from_stage,
                inputs=[job_select_pipeline, stage_select],
                outputs=[pipeline_display],
            )

        # ── Editor Tab ──
        with gr.Tab("Editor"):
            with gr.Row():
                job_select_editor = gr.Dropdown(
                    label="Select Job",
                    choices=get_job_choices(),
                    interactive=True,
                )
                transcript_type = gr.Radio(
                    label="Transcript",
                    choices=["Original Transcript", "Rewritten Transcript"],
                    value="Rewritten Transcript",
                )
                load_btn = gr.Button("Load")

            editor = gr.Textbox(
                label="Transcript Editor",
                lines=20,
                interactive=True,
                placeholder="Load a transcript to edit it here...",
            )

            with gr.Row():
                save_btn = gr.Button("Save")
                save_continue_btn = gr.Button("Save & Continue Pipeline", variant="primary")

            editor_status = gr.Textbox(label="Status", interactive=False)

            load_btn.click(
                load_transcript,
                inputs=[job_select_editor, transcript_type],
                outputs=[editor],
            )
            save_btn.click(
                save_transcript,
                inputs=[job_select_editor, transcript_type, editor],
                outputs=[editor_status],
            )
            save_continue_btn.click(
                save_and_continue,
                inputs=[job_select_editor, transcript_type, editor],
                outputs=[editor_status],
            )

        # ── Review Tab ──
        with gr.Tab("Review"):
            with gr.Row():
                job_select_review = gr.Dropdown(
                    label="Select Job",
                    choices=get_job_choices(),
                    interactive=True,
                )
                load_review_btn = gr.Button("Load Review")

            with gr.Row():
                with gr.Column(scale=2):
                    video_preview = gr.Video(label="Final Video")
                with gr.Column(scale=1):
                    thumbnail_preview = gr.Image(label="Thumbnail")

            title_input = gr.Textbox(label="Title")
            description_input = gr.Textbox(label="Description", lines=5)
            tags_input = gr.Textbox(label="Tags (comma-separated)")

            with gr.Row():
                save_meta_btn = gr.Button("Save Metadata")
                publish_btn = gr.Button("Publish to YouTube", variant="primary")

            review_status = gr.Textbox(label="Status", interactive=False)

            load_review_btn.click(
                load_review,
                inputs=[job_select_review],
                outputs=[video_preview, title_input, description_input, tags_input, thumbnail_preview],
            )
            save_meta_btn.click(
                save_metadata,
                inputs=[job_select_review, title_input, description_input, tags_input],
                outputs=[review_status],
            )
            publish_btn.click(
                publish_video,
                inputs=[job_select_review],
                outputs=[review_status],
            )

        # ── Settings Tab ──
        with gr.Tab("Settings"):
            gr.Markdown("### Model Settings")
            with gr.Row():
                whisper_model = gr.Textbox(
                    label="Whisper Model", value=config.whisper_model
                )
                llm_model = gr.Textbox(
                    label="LLM Model (Ollama)", value=config.llm_model
                )
                piper_voice = gr.Textbox(
                    label="Piper Voice", value=config.piper_voice
                )

            gr.Markdown("### Render Settings")
            with gr.Row():
                res_x = gr.Number(label="Resolution X", value=config.render_resolution_x)
                res_y = gr.Number(label="Resolution Y", value=config.render_resolution_y)
                fps = gr.Number(label="FPS", value=config.render_fps)

            gr.Markdown("### Paths")
            with gr.Row():
                workspace = gr.Textbox(
                    label="Workspace", value=str(config.workspace_dir), interactive=False
                )
                avatars = gr.Textbox(
                    label="Avatars", value=str(config.avatars_dir), interactive=False
                )

            gr.Markdown("### Avatar Files")
            avatar_list = gr.Textbox(
                label="Available Avatars",
                value="\n".join(
                    str(f.name) for f in config.avatars_dir.glob("*")
                    if f.suffix.lower() in (".glb", ".fbx", ".gltf")
                ) or "No avatars found. Place .glb or .fbx files in the avatars/ directory.",
                interactive=False,
                lines=3,
            )

    return app


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=config.web_host,
        server_port=config.web_port,
        share=False,
    )

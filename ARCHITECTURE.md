# YouTube Automation Pipeline — Architecture

## Overview

Fully automated, locally-run pipeline that takes a YouTube video link and produces
a new video with a rewritten transcript, AI-generated voice, and 3D avatar
driven by body-tracking data extracted from the original footage. Everything
runs on consumer hardware (NVIDIA 3090 / 3080 Ti) with zero paid services.

## Pipeline Stages

```
┌─────────┐   ┌────────────┐   ┌─────────┐   ┌───────────┐   ┌──────────────┐
│ Download │──▶│ Transcribe │──▶│ Rewrite │──▶│ Voice TTS │──▶│ Lip Sync Gen │
│ (yt-dlp) │   │ (Whisper)  │   │ (Ollama) │   │  (Piper)  │   │  (Rhubarb)   │
└─────────┘   └────────────┘   └─────────┘   └───────────┘   └──────────────┘
                                                                      │
┌──────────┐   ┌────────────────┐   ┌──────────────┐                  │
│ Upload   │◀──│ Blender Render │◀──│ Body Tracking │◀── original video
│ (YT API) │   │  (headless)    │   │ (GVHMR/SMPL)  │                  │
└──────────┘   └────────────────┘   └──────────────┘◀──── lip sync ───┘
                      ▲
                      │
               ┌──────────────┐
               │   Metadata   │
               │ (Ollama + SD)│
               └──────────────┘
```

### Stage Details

| # | Stage | Tool | Input | Output | GPU? |
|---|-------|------|-------|--------|------|
| 1 | **Download** | yt-dlp | YouTube URL | video.mp4 + metadata.json | No |
| 2 | **Transcribe** | faster-whisper (large-v3) | video.mp4 | transcript.json (word-level timestamps) | Yes |
| 3 | **Rewrite** | Ollama (Llama 3 8B) | transcript.json | rewritten_transcript.json | Yes |
| 4 | **Voice** | Piper TTS | rewritten_transcript.json | voice.wav | CPU (fast) |
| 5 | **Body Tracking** | GVHMR (SMPL output) + smplx | video.mp4 | tracking.json (SMPL pose params) + smpl_rig.json + smpl_mesh.npz | Yes |
| 6 | **Lip Sync** | Rhubarb Lip Sync | voice.wav | lipsync.json (phoneme timestamps) | No |
| 7 | **Blender Render** | Blender 4.x (headless, Cycles CUDA) | SMPL rig + mesh + tracking + lipsync + voice.wav | output.mp4 | GPU |
| 8 | **Metadata** | Ollama + PIL/SD | transcript + video frame | title, desc, tags, thumbnail.png | Yes |
| 9 | **Upload** | YouTube Data API v3 | output.mp4 + metadata | Published video | No |

### Key Design Decisions

- **faster-whisper** over OpenAI Whisper: 4x faster with identical accuracy, lower VRAM
- **Piper TTS** over Coqui: lightweight, consistent quality, no voice cloning needed
- **GVHMR** over MediaPipe: regresses full SMPL parameters (joint rotations
  including twist/roll, fixed bone lengths, gravity-aligned world motion)
  instead of bare landmark positions. MediaPipe only gives 3D points, which
  leaves bone twist unconstrained — catastrophic for yoga retargeting.
- **SMPL rig built in-Blender from pre-computed data**: the tracking stage
  runs `smplx` once to bake out rest joint positions, skinning weights, and
  faces; Blender's bundled Python doesn't need torch or smplx installed.
- **Rhubarb Lip Sync**: generates viseme/phoneme data from audio for 3D blend
  shapes. ⚠️ Currently a no-op at render time: the SMPL body mesh has no facial
  blend shapes (SMPL has no face). Wiring it up requires SMPL-X (FLAME face
  params) or a separate face-rigged avatar head.
- **Motion/narration sync** (`MOTION_SYNC`): the rewritten TTS narration never
  matches the source video's length, so by default the tracked motion is
  time-stretched to span the audio exactly (`stretch`). `freeze` keeps
  real-time motion and holds the last pose. The render logs a warning when
  the stretch factor falls outside 0.5–2.0.
- **Stages run sequentially**: only one model in VRAM at a time (fits 3080 Ti too)

### Required model files (one-time user actions)

GVHMR + smplx need the SMPL body model, which MPI distributes via a free
non-commercial registration:

1. Register at https://smpl.is.tue.mpg.de/
2. Download **SMPL_python_v.1.1.0** (or similar), extract `SMPL_NEUTRAL.pkl`
3. Place it at `models/smpl/SMPL_NEUTRAL.pkl` (this is volume-mounted into
   the container)

⚠️ The SMPL license is research/non-commercial. Monetized use requires a
commercial license from Meshcapade.

GVHMR also needs its pretrained checkpoints (several GB). Run
`python scripts/setup_models.py` inside the container — it downloads them
into `models/gvhmr/` (persisted volume; the image symlinks
`/opt/gvhmr/inputs/checkpoints` there) along with the Whisper, Ollama, and
Piper models.

## Job Queue & Orchestration

```
SQLite DB (jobs.db)
├── jobs table         — one row per YouTube URL
├── job_stages table   — one row per stage per job
└── queue_config       — global settings
```

- Jobs are queued with a priority and processed FIFO
- Each stage writes its output to `workspace/<job_id>/<stage_name>/`
- A stage can be: `pending`, `running`, `paused`, `completed`, `failed`, `skipped`
- **Pause/Resume**: user can pause at any stage boundary; the pipeline stops
  before starting the next stage. If paused mid-stage, that stage restarts.
- **Edit & Continue**: user edits an intermediate file (e.g., transcript),
  then resumes — the pipeline picks up from the *next* stage after the edit.

## Directory Layout

```
youtube-automation/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env.example
├── src/
│   ├── config.py              # All configuration, paths, model names
│   ├── pipeline/
│   │   ├── orchestrator.py    # Runs jobs through stages
│   │   ├── job_queue.py       # SQLite job/stage persistence
│   │   └── stage.py           # Base class for all stages
│   ├── stages/
│   │   ├── download.py        # yt-dlp wrapper
│   │   ├── transcribe.py      # faster-whisper
│   │   ├── rewrite.py         # Ollama LLM
│   │   ├── voice.py           # Piper TTS
│   │   ├── tracking.py        # GVHMR → SMPL pose params
│   │   ├── lipsync.py         # Rhubarb Lip Sync
│   │   ├── blender_render.py  # Blender headless via subprocess
│   │   ├── metadata.py        # Title/desc/tags/thumbnail
│   │   └── upload.py          # YouTube Data API v3
│   ├── blender_scripts/
│   │   └── render.py          # Builds SMPL rig, applies pose + lipsync,
│   │                          # adds audio, renders (single headless script)
│   └── ui/
│       └── app.py             # Gradio web interface
├── models/                    # Auto-downloaded model files (git-ignored)
├── workspace/                 # Job working directories (git-ignored)
├── avatars/                   # User-provided .glb/.fbx avatars (currently
│                              # unused — the avatar is built from SMPL data)
├── scripts/
│   ├── setup_models.py        # One-time model downloader
│   └── test_blender_retarget.py  # Verifies SMPL→Blender pose math
│                                 # (run inside Blender / bpy)
├── requirements.txt
└── .gitignore
```

## Web UI (Gradio)

- **Queue Tab**: paste URLs, view queue, reorder/remove
- **Pipeline Tab**: per-job stage progress, pause/resume/restart buttons
- **Editor Tab**: view/edit transcript, preview audio, re-run from edit point
- **Review Tab**: preview final video, edit metadata, publish button
- **Settings Tab**: model selection, avatar selection, voice selection, output prefs

## Deployment

- **Docker Compose** with NVIDIA Container Toolkit for GPU passthrough
- Works on Windows (WSL2) and Linux
- Single `docker compose up` to start everything
- Models auto-download on first run
- `workspace/` and `models/` are bind-mounted volumes for persistence
- Multi-machine: clone repo, `docker compose up`, done

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 Ti (12GB) | RTX 3090 (24GB) |
| RAM | 16 GB | 32 GB |
| Disk | 100 GB free | 500 GB+ |
| OS | Windows 10/11 + WSL2 | Linux (native Docker) |

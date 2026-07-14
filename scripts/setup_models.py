"""
One-time model setup script.

Downloads all required models so the pipeline can run offline.
Run this after first Docker build or on a new machine.

Usage:
    python scripts/setup_models.py
"""

import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config


def run(cmd: list[str], **kwargs):
    print(f"  → {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def setup_whisper():
    """Pre-download the Whisper model."""
    print("\n[1/5] Downloading Whisper model (large-v3)...")
    print("       This is ~3GB and may take a few minutes.")
    from faster_whisper import WhisperModel
    model = WhisperModel(config.whisper_model, device="cpu", compute_type="int8")
    del model
    print("       Done!")


def setup_ollama():
    """Pull the LLM model into the Ollama service via its HTTP API.

    The `ollama` CLI does not exist inside the app container — Ollama runs
    as a separate compose service — so we call its /api/pull endpoint.
    """
    import requests

    host = config.ollama_host
    print(f"\n[2/5] Pulling Ollama model: {config.llm_model} (via {host})")
    print("       This may take a while on first download...")
    try:
        resp = requests.post(
            f"{host}/api/pull",
            json={"name": config.llm_model},
            stream=True,
            timeout=7200,
        )
        resp.raise_for_status()
        last_status = ""
        for line in resp.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(f"Ollama pull failed: {msg['error']}")
            status = msg.get("status", "")
            if status != last_status:
                print(f"       {status}")
                last_status = status
        print("       Done!")
    except requests.ConnectionError:
        print(f"       ERROR: Could not reach Ollama at {host}.")
        print("       Start it first (docker compose up -d ollama), or pull manually:")
        print(f"       docker exec youtube-automation-ollama ollama pull {config.llm_model}")
        raise


def setup_piper():
    """Download the Piper TTS voice model."""
    print(f"\n[3/5] Downloading Piper voice: {config.piper_voice}")
    model_dir = config.models_dir / "piper"
    model_dir.mkdir(parents=True, exist_ok=True)

    voice = config.piper_voice
    parts = voice.split("-")
    lang_code = parts[0]
    lang = lang_code.split("_")[0]
    speaker = parts[1] if len(parts) > 1 else "default"
    quality = parts[2] if len(parts) > 2 else "medium"

    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    model_url = f"{base_url}/{lang}/{lang_code}/{speaker}/{quality}/{voice}.onnx"
    config_url = f"{model_url}.json"

    for url, filename in [(model_url, f"{voice}.onnx"), (config_url, f"{voice}.onnx.json")]:
        dest = model_dir / filename
        if dest.exists():
            print(f"       Already exists: {filename}")
            continue
        run(["wget", "-q", "--show-progress", "-O", str(dest), url])

    print("       Done!")


def setup_gvhmr():
    """Download GVHMR pretrained checkpoints from the official release folder.

    GVHMR's demo needs several pretrained models under
    <GVHMR_ROOT>/inputs/checkpoints/. The Docker image symlinks that path to
    /app/models/gvhmr (a mounted volume), so downloads persist across
    rebuilds. Source: the Google Drive folder linked from GVHMR's
    docs/INSTALL.md — by downloading you agree to the corresponding licenses.
    """
    print("\n[4/5] Downloading GVHMR checkpoints...")
    ckpt_root = config.models_dir / "gvhmr"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    expected = [
        ckpt_root / "gvhmr" / "gvhmr_siga24_release.ckpt",
        ckpt_root / "hmr2" / "epoch=10-step=25000.ckpt",
        ckpt_root / "vitpose" / "vitpose-h-multi-coco.pth",
        ckpt_root / "yolo" / "yolov8x.pt",
        ckpt_root / "dpvo" / "dpvo.pth",  # only needed for moving-camera videos
    ]
    missing = [p for p in expected if not p.exists()]

    if not missing:
        print("       All checkpoints already present.")
    else:
        print("       Missing:", ", ".join(str(p.relative_to(ckpt_root)) for p in missing))
        print("       Fetching official checkpoint folder (several GB)...")
        # Folder ID from GVHMR docs/INSTALL.md
        run([
            sys.executable, "-m", "gdown",
            "--folder", "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD",
            "-O", str(ckpt_root),
        ])
        still_missing = [p for p in expected if not p.exists()]
        if still_missing:
            print("       WARNING: some checkpoints are still missing "
                  "(Google Drive quota?). Download manually from the folder in "
                  "GVHMR's docs/INSTALL.md into models/gvhmr/:")
            for p in still_missing:
                print(f"         - {p.relative_to(ckpt_root)}")

    # GVHMR expects the SMPL body model at
    # inputs/checkpoints/body_models/smpl/ — link it to our models/smpl dir
    # so the user only has to place SMPL_NEUTRAL.pkl once.
    body_models = ckpt_root / "body_models"
    body_models.mkdir(parents=True, exist_ok=True)
    smpl_link = body_models / "smpl"
    if not smpl_link.exists():
        smpl_link.symlink_to(config.smpl_model_dir, target_is_directory=True)
    if not (config.smpl_model_dir / "SMPL_NEUTRAL.pkl").exists():
        print("       NOTE: models/smpl/SMPL_NEUTRAL.pkl is still missing.")
        print("       Register at https://smpl.is.tue.mpg.de and place it there.")
    print("       Done!")


def setup_avatar():
    """Check for avatar files and provide instructions."""
    print("\n[5/5] Checking for avatar files...")
    avatars = list(config.avatars_dir.glob("*.glb")) + list(config.avatars_dir.glob("*.fbx"))
    if avatars:
        print(f"       Found {len(avatars)} avatar(s):")
        for a in avatars:
            print(f"         - {a.name}")
    else:
        print("       No avatars found!")
        print(f"       Place a rigged .glb or .fbx avatar in: {config.avatars_dir}")
        print("       Recommended: Download a free rigged character from Mixamo (mixamo.com)")
        print("       - Choose a character, download as FBX with T-pose")
        print("       - The avatar should have an armature and viseme blend shapes for lip sync")


def main():
    print("=" * 60)
    print("YouTube Automation Pipeline — Model Setup")
    print("=" * 60)

    setup_whisper()
    setup_ollama()
    setup_piper()
    setup_gvhmr()
    setup_avatar()

    print("\n" + "=" * 60)
    print("Setup complete! You can now run the pipeline.")
    print("Start the UI with: python src/ui/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

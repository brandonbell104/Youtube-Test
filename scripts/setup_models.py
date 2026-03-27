"""
One-time model setup script.

Downloads all required models so the pipeline can run offline.
Run this after first Docker build or on a new machine.

Usage:
    python scripts/setup_models.py
"""

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
    print("\n[1/4] Downloading Whisper model (large-v3)...")
    print("       This is ~3GB and may take a few minutes.")
    from faster_whisper import WhisperModel
    model = WhisperModel(config.whisper_model, device="cpu", compute_type="int8")
    del model
    print("       Done!")


def setup_ollama():
    """Pull the LLM model into Ollama."""
    print(f"\n[2/4] Pulling Ollama model: {config.llm_model}")
    print("       This may take a while on first download...")
    run(["ollama", "pull", config.llm_model])
    print("       Done!")


def setup_piper():
    """Download the Piper TTS voice model."""
    print(f"\n[3/4] Downloading Piper voice: {config.piper_voice}")
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


def setup_avatar():
    """Check for avatar files and provide instructions."""
    print("\n[4/4] Checking for avatar files...")
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
    setup_avatar()

    print("\n" + "=" * 60)
    print("Setup complete! You can now run the pipeline.")
    print("Start the UI with: python src/ui/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

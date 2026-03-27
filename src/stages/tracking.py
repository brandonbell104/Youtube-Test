"""
Stage 5: Body pose tracking using MediaPipe.

Processes every frame of the video and extracts 33 body landmarks
per frame, suitable for driving a 3D avatar armature in Blender.

Outputs:
  - tracking.json   — per-frame landmark data
  - tracking_meta.json — summary (frame count, fps, etc.)
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.config import Config
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)

# MediaPipe landmark names for reference
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class TrackingStage(Stage):
    name = "tracking"

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, job_id: str, out_dir: Path) -> dict[str, Any]:
        video_path = self.prev_stage_dir(job_id, "download") / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Tracking %d frames at %.1f fps (%dx%d)", total_frames, fps, width, height
        )

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.tracking_model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self.config.tracking_min_detection_confidence,
            min_tracking_confidence=self.config.tracking_min_tracking_confidence,
        )

        frames_data = []
        frame_idx = 0
        tracked_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            frame_entry = {"frame": frame_idx, "timestamp": round(frame_idx / fps, 4)}

            if results.pose_landmarks:
                landmarks = []
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        "name": POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"landmark_{i}",
                        "x": round(lm.x, 6),
                        "y": round(lm.y, 6),
                        "z": round(lm.z, 6),
                        "visibility": round(lm.visibility, 4),
                    })
                frame_entry["landmarks"] = landmarks

                # Also store world landmarks (metric-scale 3D)
                if results.pose_world_landmarks:
                    world_landmarks = []
                    for i, lm in enumerate(results.pose_world_landmarks.landmark):
                        world_landmarks.append({
                            "name": POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"landmark_{i}",
                            "x": round(lm.x, 6),
                            "y": round(lm.y, 6),
                            "z": round(lm.z, 6),
                            "visibility": round(lm.visibility, 4),
                        })
                    frame_entry["world_landmarks"] = world_landmarks

                tracked_count += 1
            else:
                frame_entry["landmarks"] = None

            frames_data.append(frame_entry)
            frame_idx += 1

            if frame_idx % 500 == 0:
                logger.info("Tracked %d/%d frames (%.0f%%)", frame_idx, total_frames, 100 * frame_idx / max(total_frames, 1))

        cap.release()
        pose.close()

        # Write tracking data
        tracking_output = {
            "fps": fps,
            "total_frames": frame_idx,
            "tracked_frames": tracked_count,
            "resolution": {"width": width, "height": height},
            "landmark_names": POSE_LANDMARK_NAMES,
            "frames": frames_data,
        }

        output_path = out_dir / "tracking.json"
        output_path.write_text(json.dumps(tracking_output, separators=(",", ":")))

        logger.info(
            "Tracking complete: %d/%d frames with pose data (%.1f%%)",
            tracked_count,
            frame_idx,
            100 * tracked_count / max(frame_idx, 1),
        )

        return {
            "total_frames": frame_idx,
            "tracked_frames": tracked_count,
            "fps": fps,
            "tracking_rate": round(100 * tracked_count / max(frame_idx, 1), 1),
        }

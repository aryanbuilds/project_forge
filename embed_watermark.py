import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging
from watermarking import WatermarkConfig
import numpy as np
import pandas as pd
import cv2
import hashlib

DATABASE_FILE = "metadata.csv"  # Updated to match the file used in extract_watermark.py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_spectrogram(video_path: str) -> np.ndarray:
    """Generate a spectrogram from video frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    cap.release()
    spectrogram = np.mean(frames, axis=0)  # Average pixel intensity
    return spectrogram

def store_metadata(metadata: dict):
    """Store metadata in a CSV database."""
    if not Path(DATABASE_FILE).exists():
        pd.DataFrame([metadata]).to_csv(DATABASE_FILE, index=False)
    else:
        existing_data = pd.read_csv(DATABASE_FILE)
        updated_data = pd.concat([existing_data, pd.DataFrame([metadata])], ignore_index=True)
        updated_data.to_csv(DATABASE_FILE, index=False)

def overlay_spectrogram_on_video(video_path: str, spectrogram: np.ndarray, output_path: str, alpha: float):
    """Overlay the spectrogram onto the video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

    # Resize spectrogram to match video dimensions
    spectrogram_resized = cv2.resize(spectrogram, (width, height), interpolation=cv2.INTER_AREA)
    spectrogram_resized = (spectrogram_resized / 255.0).astype(np.float32)  # Normalize

    # Initialize video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Normalize frame and blend with spectrogram
        frame_float = frame.astype(np.float32) / 255.0
        blended = alpha * np.stack([spectrogram_resized] * 3, axis=2) + (1 - alpha) * frame_float
        blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

        # Write the blended frame to the output video
        out.write(blended)

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Embed Watermark into a Video's Spectrogram")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output path for the watermarked video")
    parser.add_argument("--strength", type=float, default=0.07, help="Watermark strength")
    args = parser.parse_args()

    config = WatermarkConfig(strength=args.strength)

    try:
        # Generate spectrogram from the video
        spectrogram = generate_spectrogram(args.video)

        # Generate a unique key for the watermark
        watermark_hash = hashlib.sha256(spectrogram.tobytes()).hexdigest()
        key = hashlib.sha256((watermark_hash + str(datetime.now())).encode()).hexdigest()

        # Overlay the spectrogram onto the video
        overlay_spectrogram_on_video(args.video, spectrogram, args.output, config.strength)

        # Store metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "video_path": args.video,
            "output_path": args.output,
            "strength": config.strength,
            "watermark_hash": watermark_hash,
            "key": key
        }
        store_metadata(metadata)

        print(f"Successfully embedded watermark into video. Metadata: {json.dumps(metadata, indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
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
from typing import Tuple

DATABASE_FILE = "metadata.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_spectrogram(video_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
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

    if not frames:
        raise ValueError("No frames were extracted from the video.")

    spectrogram = np.mean(frames, axis=0)  # Average pixel intensity

    # Resize spectrogram to target size if provided
    if target_size:
        spectrogram = cv2.resize(spectrogram, target_size, interpolation=cv2.INTER_AREA)

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for compatibility with .mp4 format

    # Resize spectrogram to match video dimensions
    spectrogram_resized = cv2.resize(spectrogram, (width, height), interpolation=cv2.INTER_AREA)
    spectrogram_resized = (spectrogram_resized / 255.0).astype(np.float32)  # Normalize

    # Initialize video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Failed to initialize VideoWriter. Check codec and output path.")

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
        # Open video to get dimensions
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {args.video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Log the embedding strength
        logger.info(f"Embedding watermark with strength: {config.strength}")

        # Generate spectrogram from the video
        spectrogram = generate_spectrogram(args.video, target_size=(width, height))

        # Save the spectrogram for debugging
        spectrogram_uint8 = spectrogram.astype(np.uint8)  # Ensure proper depth for saving
        cv2.imwrite("embedded_spectrogram.jpg", spectrogram_uint8)

        # Overlay the spectrogram onto the video
        overlay_spectrogram_on_video(args.video, spectrogram, args.output, config.strength)

        # Read the watermarked video file as binary
        with open(args.output, "rb") as f:
            watermarked_video_bytes = f.read()
        watermarked_video_hash = hashlib.sha256(watermarked_video_bytes).hexdigest()

        # Generate a unique key
        key = hashlib.sha256((watermarked_video_hash + str(datetime.now())).encode()).hexdigest()
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "video_path": args.video,
            "output_path": args.output,
            "strength": config.strength,
            "video_file_hash": watermarked_video_hash,
            "width": width,  # Add width to metadata
            "height": height,  # Add height to metadata
            "key": key
        }
        logger.info(f"Generated key: {key}")

        store_metadata(metadata)
        print(f"Successfully embedded watermark into video. Metadata: {json.dumps(metadata, indent=2)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
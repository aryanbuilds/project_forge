import argparse
import json
import sys
import logging
import cv2  # Replace moviepy with OpenCV for consistency
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE_FILE = "metadata.csv"  # Ensure consistency with embed_watermark.py

def verify_key(extracted_hash: str, key: str) -> bool:
    """Verify the extracted hash against the provided key."""
    if not Path(DATABASE_FILE).exists():
        raise FileNotFoundError(f"Metadata file '{DATABASE_FILE}' not found.")

    metadata = pd.read_csv(DATABASE_FILE)
    return any((metadata["watermark_hash"] == extracted_hash) & (metadata["key"] == key))

def generate_spectrogram(video_path: str, frame_interval: int = 10) -> np.ndarray:
    """Generate a spectrogram from video frames using FFT."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Select frames at regular intervals
        if frame_count % frame_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError("No frames were extracted from the video.")

    # Perform FFT on each frame and combine results
    fft_frames = [np.abs(np.fft.fft2(frame)) for frame in frames]
    spectrogram = np.mean(fft_frames, axis=0)  # Average FFT results

    # Normalize spectrogram for visualization
    spectrogram = np.log1p(spectrogram)  # Apply log scale for better visualization
    spectrogram = (spectrogram / np.max(spectrogram) * 255).astype(np.uint8)

    return spectrogram

def main():
    parser = argparse.ArgumentParser(description="Extract Watermark from a Video")
    parser.add_argument("--input", required=True, help="Watermarked video path")
    parser.add_argument("--output", required=True, help="Output path for extracted spectrogram")
    parser.add_argument("--original", required=True, help="Original video path")
    parser.add_argument("--key", required=True, help="Key associated with the video")
    args = parser.parse_args()

    try:
        # Generate spectrogram from the original video
        original_spectrogram = generate_spectrogram(args.original)

        # Generate spectrogram from the watermarked video
        watermarked_spectrogram = generate_spectrogram(args.input)

        # Extract the watermark by subtracting the original spectrogram
        extracted_spectrogram = (watermarked_spectrogram - original_spectrogram).clip(0, 255).astype(np.uint8)

        # Compute the hash of the extracted spectrogram
        extracted_hash = hashlib.sha256(extracted_spectrogram.tobytes()).hexdigest()

        # Verify the key and hash
        if verify_key(extracted_hash, args.key):
            print("Watermark verified successfully!")
        else:
            print("Verification failed. Key does not match.")

        # Save the extracted spectrogram
        cv2.imwrite(args.output, extracted_spectrogram)
        print(f"Extracted spectrogram saved to {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
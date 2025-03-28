import argparse
import sys
import logging
import hashlib
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE_FILE = "metadata.csv"

def main():
    parser = argparse.ArgumentParser(description="Verify Watermarked Video")
    parser.add_argument("--input", required=True, help="Watermarked video path")
    parser.add_argument("--key", required=True, help="Key associated with the video")
    args = parser.parse_args()

    try:
        # Compute the hash of the provided video
        with open(args.input, "rb") as f:
            input_video_hash = hashlib.sha256(f.read()).hexdigest()

        # Check metadata file
        if not Path(DATABASE_FILE).exists():
            raise FileNotFoundError(f"Metadata file '{DATABASE_FILE}' not found.")
        
        metadata = pd.read_csv(DATABASE_FILE)
        if 'key' not in metadata.columns:
            raise ValueError("Metadata file is missing 'key' column")
        if 'video_file_hash' not in metadata.columns:
            raise ValueError("Metadata file is missing 'video_file_hash' column")

        # Verify by matching the file hash and key
        row_match = metadata[
            (metadata["video_file_hash"] == input_video_hash) &
            (metadata["key"] == args.key)
        ]

        if not row_match.empty:
            logger.info("Watermark verified successfully!")
            logger.info(f"Video path: {row_match['video_path'].values[0]}")
        else:
            logger.warning("Verification failed. Hash or key not found in metadata.")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
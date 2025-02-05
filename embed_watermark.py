import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging
from watermarking import WatermarkConfig, WatermarkManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Embed Watermark into an Image")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--strength", type=float, default=0.07, help="Watermark strength")
    args = parser.parse_args()

    config = WatermarkConfig(strength=args.strength)
    manager = WatermarkManager(config)

    try:
        metadata = manager.process_image(args.input, args.output)
        print(f"Successfully embedded watermark. Metadata: {json.dumps(metadata, indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
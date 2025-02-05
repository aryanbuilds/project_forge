import argparse
import json
import sys
import logging
from watermarking import WatermarkConfig, WatermarkManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Extract Watermark from an Image")
    parser.add_argument("--input", required=True, help="Watermarked image path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--original", required=True, help="Original image path")
    args = parser.parse_args()

    config = WatermarkConfig()
    manager = WatermarkManager(config)

    try:
        result = manager.extract_watermark(args.original, args.input)
        print(f"Extraction result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
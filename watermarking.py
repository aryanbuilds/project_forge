import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import hashlib
import logging
import json
from typing import Tuple, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WatermarkConfig:
    """Configuration for watermark parameters"""
    strength: float = 0.07
    max_image_size: int = 4096     # Kept for resizing large images

class WatermarkStrategy(ABC):
    def __init__(self, config: WatermarkConfig):
        self.config = config

    @abstractmethod
    def embed(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def extract(self, original: np.ndarray, watermarked: np.ndarray) -> Tuple[np.ndarray, float]:
        pass

class SpectrogramWatermark(WatermarkStrategy):
    """Simplified strategy that uses alpha blending to embed a spectrogram."""
    def embed(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        # Convert watermark to matching size
        h, w = image.shape[:2]
        wm_resized = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_AREA)
        wm_resized = wm_resized.astype(np.float32) / 255.0  # Normalize

        # Convert image to float for blending
        img_float = image.astype(np.float32)

        # Alpha blend: new_pixel = alpha * spectrogram + (1 - alpha) * original_pixel
        alpha = self.config.strength
        blended = alpha * np.stack([wm_resized]*3, axis=2) + (1.0 - alpha) * img_float
        return np.clip(blended, 0, 255).astype(np.uint8)

    def extract(self, original: np.ndarray, watermarked: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = original.shape[:2]
        # Convert to float for reverse blending
        orig_float = original.astype(np.float32)
        wm_float = watermarked.astype(np.float32)

        alpha = self.config.strength
        # Reverse: extracted_spectrogram = (watermarked - (1 - alpha)*original) / alpha
        extracted = (wm_float - (1.0 - alpha) * orig_float) / alpha

        # Convert to grayscale shape
        extracted_gray = np.mean(extracted, axis=2)
        # Scale back to valid range
        extracted_gray = np.clip(extracted_gray, 0, 255).astype(np.uint8)
        # Use a simple confidence measure: presence of non-zero pixels
        confidence_score = float(np.mean(extracted_gray) / 255.0)

        return extracted_gray, confidence_score

class WatermarkManager:
    """Manages watermarking operations and metadata"""
    
    def __init__(self, config: WatermarkConfig):
        self.config = config
        self.strategy = SpectrogramWatermark(config)

    def process_image(self, image_path: str, output_path: str, watermark: np.ndarray) -> Dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")

            h, w = image.shape[:2]
            if h > self.config.max_image_size or w > self.config.max_image_size:
                scale = self.config.max_image_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            watermark_hash = hashlib.sha256(watermark.tobytes()).hexdigest()
            key = hashlib.sha256((watermark_hash + str(datetime.now())).encode()).hexdigest()
            logger.info(f"Generated watermark hash: {watermark_hash}")

            watermarked_image = self.strategy.embed(image, watermark)
            cv2.imwrite(output_path, watermarked_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Use PNG for lossless compression

            metadata = {
                'timestamp': datetime.now().isoformat(),
                'watermark_hash': watermark_hash,
                'original_size': image.shape,
                'config': self.config.__dict__,
                'watermark': watermark.tolist(),  # Store the actual watermark array
                'key': key
            }
            
            # Persist metadata to a JSON file
            metadata_file = Path(output_path).stem + "_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _compare_watermarks(self, stored_watermark: np.ndarray, extracted_watermark: np.ndarray) -> bool:
        """Compare stored and extracted watermarks directly"""
        if stored_watermark is None or not isinstance(stored_watermark, np.ndarray) or stored_watermark.shape != extracted_watermark.shape:
            return False
        return np.array_equal(stored_watermark, extracted_watermark)

    def extract_watermark(self, original_path: str, watermarked_path: str) -> Dict:
        try:
            original = cv2.imread(original_path)
            watermarked = cv2.imread(watermarked_path)
            if original is None or watermarked is None:
                raise ValueError("Failed to load images")

            if original.shape != watermarked.shape:
                watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)

            extracted_watermark, confidence = self.strategy.extract(original, watermarked)
            extracted_hash = hashlib.sha256(extracted_watermark.tobytes()).hexdigest()
            logger.info(f"Extracted watermark hash: {extracted_hash}")

            # Load metadata from JSON file
            metadata_file = Path(watermarked_path).stem + "_metadata.json"
            if Path(metadata_file).exists():
                with open(metadata_file, 'r') as f:
                    original_metadata = json.load(f)
                    original_metadata['watermark'] = np.array(original_metadata['watermark'])
            else:
                original_metadata = {}

            is_verified = False
            if original_metadata:
                stored_hash = original_metadata.get('watermark_hash')
                stored_watermark = np.array(original_metadata.get('watermark'))
                is_verified = stored_hash == extracted_hash and self._compare_watermarks(stored_watermark, extracted_watermark)
                # Convert back to list for JSON serialization
                original_metadata['watermark'] = stored_watermark.tolist()

            result = {
                'timestamp': datetime.now().isoformat(),
                'extracted_hash': extracted_hash,
                'confidence_score': confidence,
                'is_verified': is_verified,
                'original_metadata': original_metadata
            }

            return result

        except Exception as e:
            logger.error(f"Error extracting watermark: {str(e)}")
            raise
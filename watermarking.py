import cv2
import numpy as np
import pywt
from pathlib import Path
from datetime import datetime
import hashlib
import logging
import json
from typing import Tuple, Dict, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.fftpack import dct, idct
from concurrent.futures import ThreadPoolExecutor

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
    block_size: Tuple[int, int] = (32, 32)
    wavelet: str = 'haar'
    compression_quality: int = 100
    max_image_size: int = 4096
    min_image_size: int = 512

class WatermarkStrategy(ABC):
    def __init__(self, config: WatermarkConfig):
        self.config = config

    @abstractmethod
    def embed(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def extract(self, original: np.ndarray, watermarked: np.ndarray) -> Tuple[np.ndarray, float]:
        pass

class HybridDWTDCTWatermark(WatermarkStrategy):
    def _split_into_blocks(self, array: np.ndarray, block_size: Tuple[int, int]) -> Generator[np.ndarray, None, None]:
        """Split array into blocks of specified size"""
        h, w = array.shape
        block_h, block_w = block_size
        
        # Validate dimensions
        if h % block_h != 0 or w % block_w != 0:
            # Pad array to make it divisible by block size
            pad_h = (block_h - h % block_h) % block_h
            pad_w = (block_w - w % block_w) % block_w
            array = np.pad(array, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = array.shape
        
        # Calculate number of blocks
        n_blocks_h = h // block_h
        n_blocks_w = w // block_w
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                yield array[
                    i * block_h:(i + 1) * block_h,
                    j * block_w:(j + 1) * block_w
                ]

    def _merge_blocks(self, blocks: Generator[np.ndarray, None, None], original_shape: Tuple[int, int]) -> np.ndarray:
        """Merge blocks back into a single array"""
        h, w = original_shape
        blocks = list(blocks)  # Convert generator to a list

        if not blocks:
            raise ValueError("No blocks to merge")

        block_h, block_w = blocks[0].shape
        n_blocks_h = -(-h // block_h)  # Ceiling division
        n_blocks_w = -(-w // block_w)  # Ceiling division

        result = np.zeros((n_blocks_h * block_h, n_blocks_w * block_w), dtype=blocks[0].dtype)

        block_idx = 0
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                if block_idx < len(blocks):
                    result[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w] = blocks[block_idx]
                    block_idx += 1

        return result[:h, :w]  # Crop back to original size

    def _validate_watermark_dimensions(self, image_shape: Tuple[int, int], watermark_shape: Tuple[int, int]) -> bool:
        """Validate watermark dimensions against image dimensions"""
        img_h, img_w = image_shape[:2]
        wm_h, wm_w = watermark_shape
        
        blocks_h = img_h // self.config.block_size[0]
        blocks_w = img_w // self.config.block_size[1]
        
        return wm_h == blocks_h and wm_w == blocks_w

    def embed(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """Embed watermark using multi-threaded block processing"""
        try:
            # Validate dimensions
            if not self._validate_watermark_dimensions(image.shape, watermark.shape):
                raise ValueError("Watermark dimensions do not match image block structure")

            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            Y, Cr, Cb = cv2.split(ycrcb)
            
            # Apply DWT
            coeffs = pywt.dwt2(Y, self.config.wavelet)
            LL, (LH, HL, HH) = coeffs
            
            # Prepare blocks for parallel processing
            blocks = list(self._split_into_blocks(HL, self.config.block_size))
            watermark_blocks = list(self._split_into_blocks(watermark, self.config.block_size))
            
            # Generate PRN sequences
            prn_sequences = [
                self._generate_prn_sequence(self.config.block_size)
                for _ in range(len(blocks))
            ]
            
            # Process blocks in parallel
            with ThreadPoolExecutor() as executor:
                processed_blocks = list(executor.map(
                    self._process_block,
                    blocks,
                    watermark_blocks,
                    prn_sequences
                ))
            
            # Reconstruct HL subband
            HL_modified = self._merge_blocks(processed_blocks, HL.shape)
            
            # Inverse DWT
            Y_modified = pywt.idwt2((LL, (LH, HL_modified, HH)), self.config.wavelet)
            Y_modified = np.clip(Y_modified, 0, 255).astype(np.uint8)
            
            # Reconstruct image
            ycrcb_modified = cv2.merge([Y_modified, Cr, Cb])
            return cv2.cvtColor(ycrcb_modified, cv2.COLOR_YCrCb2BGR)
            
        except Exception as e:
            logger.error(f"Error in embed: {str(e)}")
            raise

    def _generate_prn_sequence(self, block_size: Tuple[int, int]) -> np.ndarray:
        """Generate a pseudo-random noise sequence"""
        np.random.seed(42)  # For reproducibility
        return np.random.randn(*block_size)

    def _process_block(self, block: np.ndarray, watermark_block: np.ndarray, prn: np.ndarray) -> np.ndarray:
        """Process a single block for embedding"""
        block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
        watermark_dct = block_dct + self.config.strength * watermark_block * prn
        return idct(idct(watermark_dct.T, norm='ortho').T, norm='ortho')

    def extract(self, original: np.ndarray, watermarked: np.ndarray) -> Tuple[np.ndarray, float]:
        """Extract watermark using multi-threaded block processing"""
        try:
            ycrcb_orig = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
            Y_orig, _, _ = cv2.split(ycrcb_orig)
            
            ycrcb_wm = cv2.cvtColor(watermarked, cv2.COLOR_BGR2YCrCb)
            Y_wm, _, _ = cv2.split(ycrcb_wm)
            
            # Apply DWT
            coeffs_orig = pywt.dwt2(Y_orig, self.config.wavelet)
            coeffs_wm = pywt.dwt2(Y_wm, self.config.wavelet)
            LL_orig, (LH_orig, HL_orig, HH_orig) = coeffs_orig
            LL_wm, (LH_wm, HL_wm, HH_wm) = coeffs_wm
            
            # Prepare blocks for parallel processing
            orig_blocks = list(self._split_into_blocks(HL_orig, self.config.block_size))
            wm_blocks = list(self._split_into_blocks(HL_wm, self.config.block_size))
            
            # Generate PRN sequences (same as embedding)
            prn_sequences = [
                self._generate_prn_sequence(self.config.block_size)
                for _ in range(len(orig_blocks))
            ]
            
            # Process blocks in parallel
            with ThreadPoolExecutor() as executor:
                extracted_blocks = list(executor.map(
                    self._extract_from_block,
                    orig_blocks,
                    wm_blocks,
                    prn_sequences
                ))
            
            # Merge blocks and calculate confidence
            extracted_watermark = self._merge_blocks(extracted_blocks, HL_orig.shape)
            confidence_score = self._calculate_confidence(extracted_watermark)
            
            return extracted_watermark, confidence_score
        
        except Exception as e:
            logger.error(f"Error in extract: {str(e)}")
            raise

    def _extract_from_block(self, original_block: np.ndarray, watermarked_block: np.ndarray, prn: np.ndarray) -> np.ndarray:
        """Extract watermark from a single block"""
        # Apply DCT to both blocks
        orig_dct = dct(dct(original_block.T, norm='ortho').T, norm='ortho')
        wm_dct = dct(dct(watermarked_block.T, norm='ortho').T, norm='ortho')
        
        # Extract the watermark
        extracted_values = (wm_dct - orig_dct) / self.config.strength
        # Despread using the same PRN sequence
        despread = extracted_values * prn
        
        # Threshold to get binary watermark
        return (despread > 0).astype(np.uint8)

    def _calculate_confidence(self, extracted_watermark: np.ndarray) -> float:
        """
        Calculate confidence score for extracted watermark
        Returns value between 0 and 1
        """
        # Calculate clarity of binary decisions
        threshold_distance = np.abs(extracted_watermark - 0.5)
        return float(np.mean(threshold_distance) * 2)  # Scale to 0-1 range

class WatermarkManager:
    """Manages watermarking operations and metadata"""
    
    def __init__(self, config: WatermarkConfig):
        self.config = config
        self.processor = cv2
        self.strategy = HybridDWTDCTWatermark(config)
        self.metadata_store = {}  # In production, use proper database
    
    def generate_watermark(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate appropriate size watermark based on image dimensions"""
        block_count = (
            image_shape[0] // self.config.block_size[0],
            image_shape[1] // self.config.block_size[1]
        )
        return np.random.randint(0, 2, block_count, dtype=np.uint8)
    
    def process_image(self, image_path: str, output_path: str) -> Dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")

            h, w = image.shape[:2]
            if h > self.config.max_image_size or w > self.config.max_image_size:
                scale = self.config.max_image_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            watermark = self.generate_watermark(image.shape[:2])
            watermark_hash = hashlib.sha256(watermark.tobytes()).hexdigest()
            logger.info(f"Generated watermark hash: {watermark_hash}")

            watermarked_image = self.strategy.embed(image, watermark)
            cv2.imwrite(output_path, watermarked_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Use PNG for lossless compression

            metadata = {
                'timestamp': datetime.now().isoformat(),
                'watermark_hash': watermark_hash,
                'original_size': image.shape,
                'config': self.config.__dict__,
                'watermark': watermark.tolist()  # Store the actual watermark array
            }
            
            # Persist metadata to a JSON file
            metadata_file = Path(output_path).stem + "_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

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
            else:
                original_metadata = {}

            is_verified = False
            if original_metadata:
                stored_hash = original_metadata.get('watermark_hash')
                stored_watermark = np.array(original_metadata.get('watermark'))
                is_verified = stored_hash == extracted_hash and np.array_equal(stored_watermark, extracted_watermark)

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
# Video Watermarking with Spectrogram Embedding

This project provides a robust video watermarking solution that uses spectrogram-based watermarks embedded through alpha blending. The system includes both embedding and verification capabilities.

## Setup

### Prerequisites
- Python 3.8 or higher
- OpenCV
- NumPy
- Pandas

### Installation

1. Clone the repository
2. Create and activate a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```
3. Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

### Embedding a Watermark

```sh
python embed_watermark.py --video <input_video> --output <output_video> --strength <optional_strength>
```

Parameters:
- `--video`: Path to input video file
- `--output`: Path for watermarked output video
- `--strength`: Watermark strength (default: 0.07)

The script will:
1. Generate a spectrogram from the video
2. Embed it as a watermark using alpha blending
3. Save metadata to a CSV database
4. Generate a unique verification key

### Verifying a Watermark

```sh
python extract_watermark.py --input <watermarked_video> --key <verification_key>
```

Parameters:
- `--input`: Path to watermarked video
- `--key`: Verification key received during embedding

## Testing

Run the test suite to validate:
- Tampering detection (resizing, cropping)
- Performance with different video durations
- Watermark reliability

```sh
python test_watermark.py
```

## Technical Details

- **Watermark Generation**: Creates a unique spectrogram from video frame analysis
- **Embedding Method**: Alpha blending with configurable strength
- **Verification**: Uses SHA-256 hashing and unique keys
- **Storage**: CSV-based metadata tracking
- **Supported Formats**: MP4 videos (other formats may work but are not officially supported)

## Security Features

- Tamper detection for video modifications
- Unique key generation per watermark
- Secure hash verification
- Metadata tracking for all watermarked videos

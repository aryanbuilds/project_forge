# Enterprise Image & Video Watermarking

This project provides a solution for generating a spectrogram from a video and embedding it into that video via alpha-blending.

## Setup

### 1. Create a Virtual Environment

```sh
python -m venv venv
```
### 2. Activate the Virtual Environment

- On Windows:
    ```sh
    .\venv\Scripts\activate
    ```
- On macOS/Linux:
    ```sh
    source venv/bin/activate
    ```

### 3. Install Requirements

```sh
pip install -r requirements.txt
```

## Usage

### 1. Embedding a Watermark in a Video

Use the following command to embed a spectrogram watermark:

```sh
python embed_watermark.py --video <input_video_path> --output <output_video_path> --strength <watermark_strength>
```

- --video: Path to the input video.
- --output: Path to save the watermarked video.
- --strength: (Optional) Alpha-blending strength; defaults to 0.07.

### 2. Verifying a Watermarked Video

Use the following command to verify ownership of a watermarked video:

```sh
python extract_watermark.py --input <watermarked_video_path> --key <verification_key>
```

This command computes the hash of the provided video, looks it up in your database (metadata.csv), and verifies if the key and the hashed file match an existing entry.

## Metadata

After embedding, a JSON metadata file is saved automatically (same stem as your output, plus "_metadata.json"). This metadata includes:

```json
{
  "timestamp": "2023-10-05T12:34:56.789012",
  "watermark_hash": "abcdef1234567890...",
  "original_size": [800, 600, 3],
  "config": { ...existing data... },
  "watermark": [ ... ],
  "key": "unique sha256 key"
}
```

## Additional Information

- The spectrogram is generated from the average pixel intensities of the video frames.
- The watermark is alpha-blended onto the video, then recovered during extraction.
- A unique key verifies which video the watermark belongs to, by matching the extracted hash with stored metadata.
- MoviePy (via FFmpeg) supports most popular video formats, including .mov and .mp4.

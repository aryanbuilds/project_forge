# Enterprise Image Watermarking

This project provides a robust solution for embedding and extracting watermarks in images using a hybrid DWT-DCT approach. The watermarking process is designed to be resilient and efficient, leveraging multi-threaded block processing.

## Setup

### 1. Create a Virtual Environment

To keep dependencies isolated, it's recommended to use a virtual environment.

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

Install the necessary dependencies using `requirements.txt`.

```sh
pip install -r requirements.txt
```

## Important Commands

### Embedding a Watermark

To embed a watermark into an image, use the following command:

```sh
python watermarking.py --mode embed --input <input_image_path> --output <output_image_path> --strength <watermark_strength>
```

- `--input`: Path to the input image.
- `--output`: Path to save the watermarked image.
- `--strength`: (Optional) Strength of the watermark. Default is `0.05`.

### Extracting a Watermark

To extract a watermark from a watermarked image, use the following command:

```sh
python watermarking.py --mode extract --input <watermarked_image_path> --output <output_path> --original <original_image_path>
```

- `--input`: Path to the watermarked image.
- `--output`: Path to save the extracted watermark.
- `--original`: Path to the original image.

## Additional Information

- Ensure that the images used for embedding and extraction are of the same dimensions.
- The watermarking process uses a hybrid DWT-DCT approach for robust watermark embedding and extraction.
- The project includes logging for better traceability and debugging.

For more details, refer to the source code and comments within `watermarking.py`.

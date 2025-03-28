import subprocess
import cv2
import os
from pathlib import Path

def run_command(cmd):
    process = subprocess.run(cmd.split(), capture_output=True, text=True)
    return process.returncode == 0, process.stdout, process.stderr

def test_video_tampering(input_video, output_video, key):
    """Test various video tampering scenarios"""
    print("\nRunning tampering tests...")
    
    # Test 1: Resize video
    cap = cv2.VideoCapture(output_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    resized_video = "resized_" + Path(output_video).name
    cap = cv2.VideoCapture(output_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(resized_video, fourcc, 30.0, (width//2, height//2))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (width//2, height//2))
        out.write(resized_frame)
    
    cap.release()
    out.release()
    
    # Test 2: Crop video
    cropped_video = "cropped_" + Path(output_video).name
    cap = cv2.VideoCapture(output_video)
    out = cv2.VideoWriter(cropped_video, fourcc, 30.0, (width//2, height//2))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[height//4:3*height//4, width//4:3*width//4]
        cropped_frame = cv2.resize(cropped_frame, (width//2, height//2))
        out.write(cropped_frame)
    
    cap.release()
    out.release()
    
    # Verify tampered videos
    print("\nVerifying tampered videos...")
    
    # Check resized video
    success, stdout, stderr = run_command(f"python extract_watermark.py --input {resized_video} --key {key}")
    verification_failed = "Verification failed" in (stdout + stderr)
    print(f"Resize tampering test: {'Passed' if verification_failed else 'Failed'}")
    
    # Check cropped video
    success, stdout, stderr = run_command(f"python extract_watermark.py --input {cropped_video} --key {key}")
    verification_failed = "Verification failed" in (stdout + stderr)
    print(f"Crop tampering test: {'Passed' if verification_failed else 'Failed'}")

def test_performance(video_path):
    """Test watermarking performance with different video properties"""
    print("\nRunning performance tests...")
    
    # Test different video lengths
    durations = [5, 10, 30]  # seconds
    for duration in durations:
        output = f"test_{duration}s.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames_to_keep = duration * fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, 
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        frame_count = 0
        while frame_count < frames_to_keep:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Test watermarking
        success, _, _ = run_command(f"python embed_watermark.py --video {output} --output watermarked_{output}")
        print(f"Duration test ({duration}s): {'Passed' if success else 'Failed'}")

def main():
    # Original watermarking
    input_video = "trial_video.mp4"
    output_video = "watermarked_video.mp4"
    
    success, stdout, _ = run_command(f"python embed_watermark.py --video {input_video} --output {output_video}")
    if not success:
        print("Failed to create initial watermarked video")
        return
    
    # Extract key from stdout
    import json
    metadata = json.loads(stdout.split("Metadata: ")[-1])
    key = metadata["key"]
    
    # Run tests
    test_video_tampering(input_video, output_video, key)
    test_performance(input_video)

if __name__ == "__main__":
    main()

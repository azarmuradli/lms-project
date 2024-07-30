import cv2
import os

def extract_frames(video_path, output_folder, frames_per_second):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration (seconds): {duration}")

    interval = int(fps / frames_per_second)
    count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            saved_count += 1
        count += 1

    video.release()
    print(f"Extracted {saved_count} frames from {video_path} and saved to {output_folder}")

# Example usage
video_path = "not-attention-person2.mp4"  # Replace with your video file path
output_folder = "not-attention_frames_person2"    # Replace with your desired output folder
frames_per_second = 2                 # Number of frames to extract per second
extract_frames(video_path, output_folder, frames_per_second)




import cv2
from pathlib import Path
import os # Added os for consistency, though not strictly used in _extract_frames itself

def _extract_frames(video_path: str, output_folder: str) -> list[str]:
    """
    Extracts frames from a video file, saves them as JPEGs, and returns their paths.
    One frame is extracted per second of video.

    Args:
        video_path: Path to the video file.
        output_folder: Directory to save the extracted frames.

    Returns:
        A list of file paths for the extracted frames. Returns empty list on error.
    """
    extracted_frame_paths = []
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return extracted_frame_paths

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0: # Check if fps is zero or negative
        print(f"Error: Video FPS is {fps} for file: {video_path}. Cannot extract frames.")
        video_capture.release()
        return extracted_frame_paths

    frame_cursor = 0  # Overall frame counter
    saved_frame_count = 0
    time_next_frame_capture_sec = 0.0

    # Ensure output_folder is a Path object for easy joining and creation
    output_path_obj = Path(output_folder)
    output_path_obj.mkdir(parents=True, exist_ok=True) # Ensure output_folder exists

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break # End of video or error

        current_frame_time_sec = frame_cursor / fps
        
        if current_frame_time_sec >= time_next_frame_capture_sec:
            saved_frame_count += 1
            frame_filename = f"frame_{saved_frame_count:03d}.jpg"
            # Use the Path object for constructing the full path
            frame_filepath = str(output_path_obj / frame_filename) 
            try:
                if cv2.imwrite(frame_filepath, frame):
                    extracted_frame_paths.append(frame_filepath)
                else:
                    print(f"Error: Could not save frame: {frame_filepath}")
            except Exception as e:
                print(f"Error writing frame {frame_filepath}: {e}")
            
            time_next_frame_capture_sec += 1.0 # Aim for the next second

        frame_cursor += 1

    video_capture.release()
    return extracted_frame_paths

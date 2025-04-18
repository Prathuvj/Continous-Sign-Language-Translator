import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

class FrameExtractor:
    def __init__(self, csv_path, front_video_dir, side_video_dir, output_front_dir, output_side_dir):
        """Initialize the frame extractor with paths and load CSV data."""
        self.csv_path = Path(csv_path)
        self.front_video_dir = Path(front_video_dir)
        self.side_video_dir = Path(side_video_dir)
        self.output_front_dir = Path(output_front_dir)
        self.output_side_dir = Path(output_side_dir)

        # Create output directories if they don't exist
        self.output_front_dir.mkdir(parents=True, exist_ok=True)
        self.output_side_dir.mkdir(parents=True, exist_ok=True)

        # Load CSV data with correct column names
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8', delimiter='\t')
            print(f"Successfully loaded CSV with {len(self.df)} entries")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def extract_frames_from_video(self, video_path, output_dir, start_time, end_time, clip_id):
        """Extract frames from a video between start_time and end_time with optimizations."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Calculate new dimensions (reduce to 60% of original size)
            new_width = int(width * 0.6)
            new_height = int(height * 0.6)
            
            # Create clip directory
            clip_dir = output_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Set frame position to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = 0
            total_frames = end_frame - start_frame
            
            # Calculate frame sampling rate (take every 3rd frame)
            frame_interval = 3
            
            with tqdm(total=total_frames // frame_interval, desc=f"Extracting frames for {clip_id}") as pbar:
                while cap.isOpened():
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if current_frame >= end_frame:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Only process every nth frame
                    if frame_count % frame_interval != 0:
                        frame_count += 1
                        continue

                    # Resize frame
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Save frame with high compression
                    frame_path = clip_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    
                    frame_count += 1
                    pbar.update(1)

            cap.release()
            print(f"Extracted {frame_count // frame_interval} frames for clip {clip_id}")

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return False

        return True

    def process_videos(self):
        """Process all videos listed in the CSV file."""
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing clips"):
            try:
                # Extract relevant information from row using correct column names
                video_id = row['VIDEO_ID']
                video_name = row['VIDEO_NAME']  # This contains the full video name
                sentence_id = row['SENTENCE_ID']
                start_time = float(row['START_REALIGNED'])
                end_time = float(row['END_REALIGNED'])

                # --- Check if frames already exist --- 
                front_clip_dir = self.output_front_dir / sentence_id
                side_clip_dir = self.output_side_dir / sentence_id

                # Check if both front and side directories exist and are not empty (or only front if side doesn't exist)
                front_exists_and_populated = front_clip_dir.exists() and any(front_clip_dir.iterdir())
                
                # Construct potential side video path to see if side video should exist
                parts = video_name.split('-rgb_front')
                if len(parts) > 0:
                    base_id = parts[0]
                    side_video_path_check = self.side_video_dir / f"{base_id}-rgb_side.mp4"
                    side_should_exist = side_video_path_check.exists()
                else:
                    side_should_exist = False # Cannot determine if side video should exist

                side_exists_and_populated = side_clip_dir.exists() and any(side_clip_dir.iterdir())

                # Determine if we need to skip based on existing frames
                skip_clip = False
                if side_should_exist:
                    if front_exists_and_populated and side_exists_and_populated:
                        skip_clip = True
                elif front_exists_and_populated:
                    # If side shouldn't exist, only check front
                    skip_clip = True
                
                if skip_clip:
                    # print(f"Skipping clip {sentence_id}, frames already extracted.") # Optional: uncomment for verbose output
                    continue
                # --- End Check --- 

                # Get the number from video_name (e.g., -5- or -8-)
                parts = video_name.split('-rgb_front')
                if len(parts) > 0:
                    base_id = parts[0]  # This will be like "videoID-5"
                    
                    # Process front view video
                    front_video_path = self.front_video_dir / f"{base_id}-rgb_front.mp4"
                    if front_video_path.exists():
                        if not front_exists_and_populated: # Only extract if not already done
                            self.extract_frames_from_video(
                                front_video_path,
                                self.output_front_dir,
                                start_time,
                                end_time,
                                sentence_id
                            )
                    else:
                        print(f"Front video not found: {front_video_path}")

                    # Process side view video
                    side_video_path = self.side_video_dir / f"{base_id}-rgb_side.mp4"
                    
                    if side_video_path.exists():
                        if not side_exists_and_populated: # Only extract if not already done
                            self.extract_frames_from_video(
                                side_video_path,
                                self.output_side_dir,
                                start_time,
                                end_time,
                                sentence_id
                            )
                    else:
                        # No need to print 'not found' if it's not supposed to exist
                        if side_should_exist:
                            print(f"Side video not found: {side_video_path}")

            except Exception as e:
                print(f"Error processing row {row.get('SENTENCE_ID', 'unknown')}: {e}")
                continue

def main():
    """Main function to run the frame extractor."""
    # Configure paths
    base_dir = Path.cwd()
    csv_path = base_dir / "how2sign_realigned_train.csv"
    front_video_dir = base_dir / "train_raw_videos"
    side_video_dir = base_dir / "train_side_raw_videos"
    output_front_dir = base_dir / "data/extracted_frames/front"
    output_side_dir = base_dir / "data/extracted_frames/side"

    try:
        extractor = FrameExtractor(
            csv_path=csv_path,
            front_video_dir=front_video_dir,
            side_video_dir=side_video_dir,
            output_front_dir=output_front_dir,
            output_side_dir=output_side_dir
        )
        extractor.process_videos()
    except Exception as e:
        print(f"Error running frame extractor: {e}")
        raise

if __name__ == "__main__":
    main() 
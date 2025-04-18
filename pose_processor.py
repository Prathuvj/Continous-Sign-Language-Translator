import cv2
import mediapipe as mp
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import os
import signal
import pickle
from typing import List, Dict, Any, Optional
import platform

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

def process_frame(frame_path: Path, holistic: mp.solutions.holistic.Holistic) -> tuple:
    """Process a single frame"""
    try:
        # Read the frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logging.warning(f"Could not read frame: {frame_path}")
            return frame_path, None
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        mp_results = holistic.process(frame_rgb)
        
        # Extract landmarks
        landmarks = {
            'pose': extract_pose_landmarks(mp_results.pose_landmarks),
            'face': extract_face_landmarks(mp_results.face_landmarks),
            'left_hand': extract_hand_landmarks(mp_results.left_hand_landmarks),
            'right_hand': extract_hand_landmarks(mp_results.right_hand_landmarks)
        }
        
        return frame_path, landmarks
        
    except Exception as e:
        logging.error(f"Error processing frame {frame_path}: {str(e)}")
        return frame_path, None

def extract_pose_landmarks(landmarks):
    """Extract pose landmarks"""
    if landmarks is None:
        return None
        
    return [
        {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        }
        for landmark in landmarks.landmark
    ]

def extract_face_landmarks(landmarks):
    """Extract face landmarks"""
    if landmarks is None:
        return None
        
    return [
        {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        }
        for landmark in landmarks.landmark
    ]

def extract_hand_landmarks(landmarks):
    """Extract hand landmarks"""
    if landmarks is None:
        return None
        
    return [
        {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        }
        for landmark in landmarks.landmark
    ]

class PoseProcessor:
    def __init__(self, frames_dir: str, output_dir: str, frame_skip: int = 1, model_complexity: int = 1):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_skip = frame_skip
        self.model_complexity = model_complexity
        self.killer = GracefulKiller()
        
        # Create checkpoint file
        self.checkpoint_file = self.output_dir / 'checkpoint.pkl'
        self.processed_segments = self._load_checkpoint()
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        logging.info(f"Initialized PoseProcessor with frame_skip={frame_skip}")
        
    def _load_checkpoint(self) -> set:
        """Load processed segments from checkpoint file"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
        return set()
    
    def _save_checkpoint(self):
        """Save processed segments to checkpoint file"""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.processed_segments, f)
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
    
    def process_video_segment(self, segment_id: str):
        """Process all frames in a video segment"""
        if segment_id in self.processed_segments:
            logging.info(f"Skipping already processed segment: {segment_id}")
            return
            
        segment_dir = self.frames_dir / segment_id
        if not segment_dir.exists():
            logging.warning(f"Segment directory not found: {segment_dir}")
            return
            
        # Create output directory for this segment
        output_segment_dir = self.output_dir / segment_id
        output_segment_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all frame files and apply frame skipping
        frame_files = sorted(list(segment_dir.glob("*.jpg")))[::self.frame_skip]
        if not frame_files:
            logging.warning(f"No frames found in {segment_dir}")
            return
            
        logging.info(f"Processing {len(frame_files)} frames for segment {segment_id}")
        
        try:
            # Process frames sequentially
            for frame_path in tqdm(frame_files, desc=f"Processing {segment_id}", unit="frame"):
                if self.killer.kill_now:
                    break
                    
                frame_path, landmarks = process_frame(frame_path, self.holistic)
                if landmarks is not None:
                    output_file = output_segment_dir / f"{frame_path.stem}_landmarks.json"
                    with open(output_file, 'w') as f:
                        json.dump(landmarks, f, indent=2)
            
            if not self.killer.kill_now:
                self.processed_segments.add(segment_id)
                self._save_checkpoint()
                
        except Exception as e:
            logging.error(f"Error processing segment {segment_id}: {str(e)}")
                    
    def process_all_segments(self):
        """Process all video segments in the frames directory"""
        segment_dirs = [d for d in self.frames_dir.iterdir() if d.is_dir()]
        total_segments = len(segment_dirs)
        
        logging.info(f"Found {total_segments} segments to process")
        
        for segment_dir in tqdm(segment_dirs, desc="Processing segments", unit="segment"):
            if self.killer.kill_now:
                logging.info("Gracefully shutting down...")
                break
            self.process_video_segment(segment_dir.name)
        
        # Clean up MediaPipe resources
        self.holistic.close()

def process_view(view: str, frame_skip: int = 1, model_complexity: int = 1):
    """Process frames for a specific view (front or side)"""
    logging.info(f"Starting pose processing for {view} view...")
    
    # Set up paths
    frames_dir = f"data/extracted_frames/{view}"
    output_dir = f"data/pose_landmarks/{view}"
    
    # Create and run the processor with optimizations
    processor = PoseProcessor(
        frames_dir=frames_dir,
        output_dir=output_dir,
        frame_skip=frame_skip,
        model_complexity=model_complexity
    )
    
    try:
        processor.process_all_segments()
        logging.info(f"Pose processing completed for {view} view!")
    except KeyboardInterrupt:
        logging.info("Process interrupted by user. Saving checkpoint...")
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
    finally:
        processor._save_checkpoint()

def main():
    # Set Windows-specific environment variables for MediaPipe
    if platform.system() == "Windows":
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
        os.environ["PYTHONPATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    # Process both views with optimizations
    for view in ["front", "side"]:
        process_view(
            view=view,
            frame_skip=2,  # Process every 2nd frame
            model_complexity=1  # Reduced model complexity
        )

if __name__ == "__main__":
    main() 
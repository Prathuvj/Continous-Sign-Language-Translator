import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import os
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('landmark_normalization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class LandmarkNormalizer:
    def __init__(self, input_dir: str, output_dir: str, batch_size: int = 1000, num_workers: int = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        # Use 75% of available CPU cores
        self.num_workers = num_workers or max(1, int(mp.cpu_count() * 0.75))
        
        # MediaPipe landmark indices for key reference points
        self.REFERENCE_POINTS = {
            'shoulder_center': 11,  # Left shoulder
            'hip_center': 23,       # Left hip
            'nose': 0,              # Nose tip
        }
        
        logging.info(f"Initialized LandmarkNormalizer with {self.num_workers} workers")
        
    def normalize_landmarks_batch(self, landmarks_batch: List[Dict]) -> List[Dict]:
        """Normalize a batch of landmarks"""
        normalized_batch = []
        for landmarks in landmarks_batch:
            if not landmarks or 'pose' not in landmarks:
                continue
                
            # Get reference points
            ref_points = {}
            for name, idx in self.REFERENCE_POINTS.items():
                if landmarks['pose'] and idx < len(landmarks['pose']):
                    landmark = landmarks['pose'][idx]
                    ref_points[name] = np.array([
                        landmark['x'],
                        landmark['y'],
                        landmark['z']
                    ])
            
            if not ref_points:
                continue
                
            # Calculate normalization factors
            if 'shoulder_center' not in ref_points or 'hip_center' not in ref_points:
                continue
                
            shoulder = ref_points['shoulder_center']
            hip = ref_points['hip_center']
            scale = np.linalg.norm(shoulder - hip)
            
            if scale == 0:
                continue
                
            translation = ref_points.get('nose', np.zeros(3))
            
            # Normalize landmarks
            normalized = {}
            for landmark_type in ['pose', 'face', 'left_hand', 'right_hand']:
                if landmarks.get(landmark_type):
                    normalized[landmark_type] = [
                        {
                            'x': (landmark['x'] - translation[0]) / scale,
                            'y': (landmark['y'] - translation[1]) / scale,
                            'z': (landmark['z'] - translation[2]) / scale,
                            **({'visibility': landmark['visibility']} if 'visibility' in landmark else {})
                        }
                        for landmark in landmarks[landmark_type]
                    ]
            
            normalized_batch.append(normalized)
            
        return normalized_batch
        
    def process_segment(self, segment_id: str):
        """Process all landmark files in a segment"""
        input_segment_dir = self.input_dir / segment_id
        output_segment_dir = self.output_dir / segment_id
        output_segment_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all landmark files
        landmark_files = sorted(list(input_segment_dir.glob("*_landmarks.json")))
        if not landmark_files:
            return
            
        # Process in batches
        total_files = len(landmark_files)
        num_batches = (total_files + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_files)
            batch_files = landmark_files[start_idx:end_idx]
            
            # Load batch of landmarks
            landmarks_batch = []
            for file_path in batch_files:
                try:
                    with open(file_path, 'r') as f:
                        landmarks_batch.append(json.load(f))
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {str(e)}")
                    continue
            
            # Normalize batch
            normalized_batch = self.normalize_landmarks_batch(landmarks_batch)
            
            # Save normalized landmarks
            for file_path, normalized in zip(batch_files, normalized_batch):
                if normalized:
                    output_file = output_segment_dir / f"{file_path.stem}_normalized.json"
                    try:
                        with open(output_file, 'w') as f:
                            json.dump(normalized, f)
                    except Exception as e:
                        logging.error(f"Error saving {output_file}: {str(e)}")
                        
    def process_all_segments(self):
        """Process all segments in parallel"""
        segment_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        total_segments = len(segment_dirs)
        
        logging.info(f"Found {total_segments} segments to process")
        
        # Process segments in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            list(tqdm(
                executor.map(self.process_segment, [d.name for d in segment_dirs]),
                total=total_segments,
                desc="Processing segments",
                unit="segment"
            ))

def process_view(view: str, batch_size: int = 1000, num_workers: int = None):
    """Process landmarks for a specific view (front or side)"""
    start_time = time.time()
    logging.info(f"Starting landmark normalization for {view} view...")
    
    # Set up paths
    input_dir = f"data/pose_landmarks/{view}"
    output_dir = f"data/normalized_landmarks/{view}"
    
    # Create and run the normalizer
    normalizer = LandmarkNormalizer(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    try:
        normalizer.process_all_segments()
        elapsed_time = time.time() - start_time
        logging.info(f"Landmark normalization completed for {view} view in {elapsed_time:.2f} seconds!")
    except Exception as e:
        logging.error(f"Error during normalization: {str(e)}")

def main():
    # Process both views with optimized parameters
    for view in ["front", "side"]:
        process_view(
            view=view,
            batch_size=2000,  # Increased batch size
            num_workers=None  # Auto-detect optimal number of workers
        )

if __name__ == "__main__":
    main() 
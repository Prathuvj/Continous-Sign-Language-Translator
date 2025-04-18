import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys
from typing import Dict, List, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('consolidation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class LandmarkConsolidator:
    def __init__(self, input_dir: str, output_file: str, num_workers: int = None, chunk_size: int = 1000):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.num_workers = num_workers or max(1, int(mp.cpu_count() * 0.75))
        self.chunk_size = chunk_size
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Define landmark types and their expected counts
        self.landmark_counts = {
            'pose': 33,
            'face': 468,
            'left_hand': 21,
            'right_hand': 21
        }
        
        # Generate column names once
        self.columns = ['segment_id', 'frame_name']
        for landmark_type in self.landmark_counts.keys():
            for i in range(self.landmark_counts[landmark_type]):
                for coord in ['x', 'y', 'z']:
                    self.columns.append(f'{landmark_type}_{i}_{coord}')
                if landmark_type == 'pose':
                    self.columns.append(f'{landmark_type}_{i}_visibility')
        
        logging.info(f"Initialized LandmarkConsolidator with {self.num_workers} workers")
        
    def flatten_landmarks(self, landmarks: Dict, segment_id: str, frame_name: str) -> Dict:
        """Flatten landmark data into a single row"""
        flattened = {
            'segment_id': segment_id,
            'frame_name': frame_name
        }
        
        # Process each landmark type
        for landmark_type, points in landmarks.items():
            if points is None:
                # Fill with zeros if landmarks are missing
                for i in range(self.landmark_counts[landmark_type]):
                    for coord in ['x', 'y', 'z']:
                        flattened[f'{landmark_type}_{i}_{coord}'] = 0
                    if landmark_type == 'pose':
                        flattened[f'{landmark_type}_{i}_visibility'] = 0
            else:
                # Add each coordinate
                for i, point in enumerate(points):
                    for coord in ['x', 'y', 'z']:
                        flattened[f'{landmark_type}_{i}_{coord}'] = point[coord]
                    if landmark_type == 'pose':
                        flattened[f'{landmark_type}_{i}_visibility'] = point.get('visibility', 0)
        
        return flattened
        
    def process_segment(self, segment_path: Path) -> List[Dict]:
        """Process all landmark files in a segment"""
        try:
            landmark_files = sorted(list(segment_path.glob("*.json")))
            rows = []
            
            for landmark_file in landmark_files:
                try:
                    with open(landmark_file, 'r') as f:
                        landmarks = json.load(f)
                    
                    # Extract segment_id and frame_name from file path
                    segment_id = segment_path.name
                    frame_name = landmark_file.stem.replace('_landmarks', '').replace('_normalized', '')
                    
                    # Flatten landmarks into a row
                    row = self.flatten_landmarks(landmarks, segment_id, frame_name)
                    rows.append(row)
                    
                except Exception as e:
                    logging.error(f"Error processing file {landmark_file}: {str(e)}")
                    continue
                    
            return rows
            
        except Exception as e:
            logging.error(f"Error processing segment {segment_path}: {str(e)}")
            return []
    
    def write_chunk_to_csv(self, chunk: List[Dict], is_first_chunk: bool):
        """Write a chunk of data to CSV file"""
        mode = 'w' if is_first_chunk else 'a'
        with open(self.output_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            if is_first_chunk:
                writer.writeheader()
            writer.writerows(chunk)
            
    def consolidate_landmarks(self):
        """Consolidate all landmark files into a single CSV"""
        logging.info(f"Starting landmark consolidation from {self.input_dir}")
        
        # Get all segment directories
        segment_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        total_segments = len(segment_dirs)
        logging.info(f"Found {total_segments} segments to process")
        
        # Process segments in chunks
        current_chunk = []
        total_rows = 0
        is_first_chunk = True
        
        # Process segments in parallel by chunks
        for i in range(0, len(segment_dirs), self.chunk_size):
            chunk_dirs = segment_dirs[i:i + self.chunk_size]
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.process_segment, d) for d in chunk_dirs]
                
                # Process results as they complete
                for future in tqdm(futures, total=len(futures), 
                                 desc=f"Processing segments {i}-{min(i+self.chunk_size, total_segments)}"):
                    rows = future.result()
                    if rows:
                        self.write_chunk_to_csv(rows, is_first_chunk)
                        total_rows += len(rows)
                        is_first_chunk = False
        
        logging.info(f"Saved consolidated landmarks to {self.output_file}")
        logging.info(f"Total frames processed: {total_rows}")

def process_landmarks(landmark_type: str):
    """Process either normal or normalized landmarks"""
    input_dir = f"data/{landmark_type}/front"  # Start with front view
    output_file = f"data/consolidated/{landmark_type}_front.csv"
    
    consolidator = LandmarkConsolidator(input_dir, output_file, chunk_size=100)
    consolidator.consolidate_landmarks()
    
    # Process side view
    input_dir = f"data/{landmark_type}/side"
    output_file = f"data/consolidated/{landmark_type}_side.csv"
    
    consolidator = LandmarkConsolidator(input_dir, output_file, chunk_size=100)
    consolidator.consolidate_landmarks()

def main():
    # Create output directory
    Path("data/consolidated").mkdir(parents=True, exist_ok=True)
    
    # Process both types of landmarks
    for landmark_type in ["pose_landmarks", "normalized_landmarks"]:
        logging.info(f"Processing {landmark_type}...")
        process_landmarks(landmark_type)
        logging.info(f"Completed processing {landmark_type}")

if __name__ == "__main__":
    main() 
import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class FrameProcessor:
    def __init__(self, json_dir, front_frames_dir, side_frames_dir):
        self.json_dir = Path(json_dir)
        self.front_frames_dir = Path(front_frames_dir)
        self.side_frames_dir = Path(side_frames_dir)
        
        # Create output directory if it doesn't exist
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized FrameProcessor with:")
        logging.info(f"  JSON directory: {self.json_dir}")
        logging.info(f"  Front frames directory: {self.front_frames_dir}")
        logging.info(f"  Side frames directory: {self.side_frames_dir}")
        
    def count_frames(self, sentence_id, is_front=True):
        """Count frames in a video directory"""
        base_dir = self.front_frames_dir if is_front else self.side_frames_dir
        segment_dir = base_dir / sentence_id
        
        if not segment_dir.exists():
            logging.warning(f"Directory not found: {segment_dir}")
            return 0
            
        try:
            # Count all .jpg files in the directory
            frame_count = len([f for f in segment_dir.glob("*.jpg")])
            logging.debug(f"Found {frame_count} frames in {segment_dir}")
            return frame_count
        except Exception as e:
            logging.error(f"Error counting frames in {segment_dir}: {str(e)}")
            return 0
            
    def process_json_files(self):
        """Process all JSON files and update frame counts"""
        json_files = list(self.json_dir.glob("*_alignment.json"))
        total_files = len(json_files)
        logging.info(f"Found {total_files} JSON files to process")
        
        for json_file in tqdm(json_files, desc="Processing JSON files", unit="file", file=sys.stdout):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Use the full sentence_id for directory lookup
                sentence_id = data['sentence_id']
                
                # Count frames for both front and side views
                front_count = self.count_frames(sentence_id, is_front=True)
                side_count = self.count_frames(sentence_id, is_front=False)
                
                # Update the frame counts in the JSON data
                data['frame_count'] = {
                    'front': front_count,
                    'side': side_count
                }
                
                # Save the updated JSON file
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                logging.info(f"Processed {json_file.name}: {front_count} front frames, {side_count} side frames")
                    
            except Exception as e:
                logging.error(f"Error processing JSON file {json_file}: {str(e)}")
                continue

def main():
    logging.info("Starting frame processing...")
    
    # Set up paths
    json_dir = "data/processed_text"
    front_frames_dir = "data/extracted_frames/front"
    side_frames_dir = "data/extracted_frames/side"
    
    # Create and run the processor
    processor = FrameProcessor(json_dir, front_frames_dir, side_frames_dir)
    processor.process_json_files()
    
    logging.info("Frame processing completed!")

if __name__ == "__main__":
    main() 
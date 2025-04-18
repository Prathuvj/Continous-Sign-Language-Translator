import pandas as pd
import re
from pathlib import Path
import json
from tqdm import tqdm
import unicodedata
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_processing.log'),
        logging.StreamHandler()
    ]
)

class TextProcessor:
    def __init__(self, csv_path, output_dir):
        """Initialize the text processor with paths and load CSV data."""
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Loading CSV from {csv_path}")
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8', delimiter='\t')
            logging.info(f"Successfully loaded CSV with {len(self.df)} entries")
        except Exception as e:
            logging.error(f"Error loading CSV: {e}")
            raise

    def clean_text(self, text):
        """Clean and normalize the text transcript."""
        if not isinstance(text, str):
            logging.warning(f"Received non-string input: {text}")
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove punctuation except apostrophes for contractions
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def process_transcripts(self):
        """Process all transcripts and create alignment files."""
        total_processed = 0
        total_errors = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing transcripts"):
            try:
                sentence_id = row['SENTENCE_ID']
                transcript = row['SENTENCE']
                start_time = float(row['START_REALIGNED'])
                end_time = float(row['END_REALIGNED'])
                
                # Clean the transcript
                cleaned_transcript = self.clean_text(transcript)
                
                # Create alignment data
                alignment_data = {
                    'sentence_id': sentence_id,
                    'original_transcript': transcript,
                    'cleaned_transcript': cleaned_transcript,
                    'start_time': start_time,
                    'end_time': end_time,
                    'frame_count': 0
                }
                
                # Save alignment data
                output_file = self.output_dir / f"{sentence_id}_alignment.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(alignment_data, f, ensure_ascii=False, indent=2)
                
                total_processed += 1
                if total_processed % 100 == 0:
                    logging.info(f"Processed {total_processed} transcripts")
                
            except Exception as e:
                total_errors += 1
                logging.error(f"Error processing transcript {row.get('SENTENCE_ID', 'unknown')}: {e}")
                continue
        
        logging.info(f"Processing completed. Total processed: {total_processed}, Errors: {total_errors}")

def main():
    """Main function to run the text processor."""
    try:
        # Configure paths
        base_dir = Path.cwd()
        csv_path = base_dir / "how2sign_realigned_train.csv"
        output_dir = base_dir / "data/processed_text"
        
        logging.info("Starting text processing")
        logging.info(f"CSV path: {csv_path}")
        logging.info(f"Output directory: {output_dir}")
        
        processor = TextProcessor(
            csv_path=csv_path,
            output_dir=output_dir
        )
        processor.process_transcripts()
        logging.info("Text processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Error running text processor: {e}")
        raise

if __name__ == "__main__":
    main() 
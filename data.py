import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SignLanguageDataset(Dataset):
    """Dataset for sign language translation"""
    
    def __init__(
        self,
        landmarks_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        max_seq_length: int = 300,
        stride: int = 150,
        front_view: bool = True,
        side_view: bool = True,
        transform=None,
        csv_delimiter: str = '\t',
        video_id_column: str = 'VIDEO_ID',
        text_column: str = 'SENTENCE',
        chunk_size: int = 1000  # Number of rows to load at once
    ):
        """
        Args:
            landmarks_dir: Directory containing normalized landmarks
            annotations_file: Path to the annotations CSV file
            max_seq_length: Maximum sequence length for sliding window
            stride: Stride for sliding window
            front_view: Whether to use front view landmarks
            side_view: Whether to use side view landmarks
            transform: Optional transform to be applied on landmarks
            csv_delimiter: Delimiter used in CSV files
            video_id_column: Column name for video IDs
            text_column: Column name for text data
            chunk_size: Number of rows to load at once for memory efficiency
        """
        self.landmarks_dir = Path(landmarks_dir)
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.front_view = front_view
        self.side_view = side_view
        self.transform = transform
        self.video_id_column = video_id_column
        self.text_column = text_column
        self.chunk_size = chunk_size
        
        # Load annotations with correct parameters
        try:
            self.annotations = pd.read_csv(
                annotations_file,
                delimiter=csv_delimiter,
                encoding='utf-8',
                on_bad_lines='skip'
            )
        except Exception as e:
            raise ValueError(f"Error reading annotations file {annotations_file}: {str(e)}")
        
        # Ensure required columns exist
        required_columns = [video_id_column, text_column]
        missing_columns = [col for col in required_columns if col not in self.annotations.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in annotations file: {missing_columns}\n"
                f"Available columns: {self.annotations.columns.tolist()}"
            )
        
        # Set up landmark file paths
        self.front_landmarks_path = self.landmarks_dir / "consolidated" / "normalized_landmarks_front.csv"
        self.side_landmarks_path = self.landmarks_dir / "consolidated" / "normalized_landmarks_side.csv"
        
        if front_view and not self.front_landmarks_path.exists():
            raise FileNotFoundError(f"Front landmarks file not found at {self.front_landmarks_path}")
        if side_view and not self.side_landmarks_path.exists():
            raise FileNotFoundError(f"Side landmarks file not found at {self.side_landmarks_path}")
        
        # Get landmark column names
        if front_view:
            try:
                # Read just the header to get column names
                first_chunk = pd.read_csv(self.front_landmarks_path, nrows=0)
                self.front_landmark_columns = [col for col in first_chunk.columns if any(
                    prefix in col for prefix in ['pose_', 'face_', 'left_hand_', 'right_hand_']
                )]
            except Exception as e:
                raise ValueError(f"Error reading front landmarks: {str(e)}")
                
        if side_view:
            try:
                # Read just the header to get column names
                first_chunk = pd.read_csv(self.side_landmarks_path, nrows=0)
                self.side_landmark_columns = [col for col in first_chunk.columns if any(
                    prefix in col for prefix in ['pose_', 'face_', 'left_hand_', 'right_hand_']
                )]
            except Exception as e:
                raise ValueError(f"Error reading side landmarks: {str(e)}")
        
        # Create sequence windows
        self.windows = self._create_sequence_windows()
        
    def _create_sequence_windows(self) -> List[Dict]:
        """Create sliding windows over the sequences"""
        windows = []
        
        for _, row in self.annotations.iterrows():
            video_id = row[self.video_id_column]
            text = row[self.text_column]
            
            # Create a window reference (actual data will be loaded during __getitem__)
            window = {
                'video_id': video_id,
                'text': text,
                'chunk_index': len(windows)  # Keep track of position for chunked loading
            }
            
            windows.append(window)
                    
        return windows
        
    def _load_chunk_data(self, video_id: str, landmarks_path: Path, landmark_columns: List[str]) -> pd.DataFrame:
        """Load data for a specific video_id from chunks"""
        video_data = []
        
        # Read the file in chunks, using segment_id for landmarks files
        chunks = pd.read_csv(
            landmarks_path,
            chunksize=self.chunk_size,
            usecols=['segment_id'] + landmark_columns,
            dtype={
                'segment_id': str,
                **{col: np.float32 for col in landmark_columns}
            }
        )
        
        for chunk in chunks:
            # Extract video ID from segment_id (format: videoID_frameNumber)
            video_chunk = chunk[chunk['segment_id'].str.split('_').str[0] == video_id]
            if len(video_chunk) > 0:
                video_data.append(video_chunk[landmark_columns])
            
        if not video_data:
            return None
            
        return pd.concat(video_data, axis=0)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sequence window"""
        window = self.windows[idx]
        video_id = window['video_id']
        
        # Load landmark data for this video
        front_data = None
        side_data = None
        
        if self.front_view:
            front_data = self._load_chunk_data(
                video_id,
                self.front_landmarks_path,
                self.front_landmark_columns
            )
            
        if self.side_view:
            side_data = self._load_chunk_data(
                video_id,
                self.side_landmarks_path,
                self.side_landmark_columns
            )
        
        # Convert to tensors with consistent dimensions
        if front_data is not None and len(front_data) > 0:
            front_tensor = torch.tensor(front_data.values, dtype=torch.float32)
        else:
            # Create empty tensor with correct feature dimension
            front_tensor = torch.zeros((0, len(self.front_landmark_columns)), dtype=torch.float32)
            
        if side_data is not None and len(side_data) > 0:
            side_tensor = torch.tensor(side_data.values, dtype=torch.float32)
        else:
            # Create empty tensor with correct feature dimension
            side_tensor = torch.zeros((0, len(self.side_landmark_columns)), dtype=torch.float32)
        
        if self.transform:
            if front_tensor.numel() > 0:
                front_tensor = self.transform(front_tensor)
            if side_tensor.numel() > 0:
                side_tensor = self.transform(side_tensor)
        
        return {
            'video_id': video_id,
            'text': window['text'],
            'front_seq': front_tensor,
            'side_seq': side_tensor,
            'seq_length': max(
                front_tensor.size(0),
                side_tensor.size(0)
            )
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    # Get max sequence length in this batch
    max_front_len = max(item['front_seq'].size(0) for item in batch)
    max_side_len = max(item['side_seq'].size(0) for item in batch) if batch[0]['side_seq'] is not None else 0
    
    # Prepare tensors
    batch_size = len(batch)
    front_seqs = []
    side_seqs = []
    seq_lengths = []
    target_seqs = []
    
    for item in batch:
        # Pad front sequence
        front_seq = item['front_seq']
        pad_length = max_front_len - front_seq.size(0)
        if pad_length > 0:
            front_seq = torch.nn.functional.pad(front_seq, (0, 0, 0, pad_length))
        front_seqs.append(front_seq)
        
        # Pad side sequence if exists
        if item['side_seq'] is not None:
            side_seq = item['side_seq']
            pad_length = max_side_len - side_seq.size(0)
            if pad_length > 0:
                side_seq = torch.nn.functional.pad(side_seq, (0, 0, 0, pad_length))
            side_seqs.append(side_seq)
        
        seq_lengths.append(item['seq_length'])
        target_seqs.append(item['text'])
    
    # Stack tensors
    front_sequences = torch.stack(front_seqs)
    side_sequences = torch.stack(side_seqs) if side_seqs else None
    seq_lengths = torch.tensor(seq_lengths)
    target_sequences = torch.tensor(target_seqs)
    
    # Create padding mask
    target_padding_mask = (target_sequences == 0)
    
    return {
        'front_sequences': front_sequences,
        'side_sequences': side_sequences,
        'seq_lengths': seq_lengths,
        'target_sequences': target_sequences,
        'target_padding_mask': target_padding_mask
    }

class SignLanguageDataModule:
    """Data module for sign language translation"""
    def __init__(self, data_config, model_config):
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self):
        """Set up the datasets"""
        self.train_dataset = SignLanguageDataset(
            landmarks_dir=self.data_config.data_dir,
            annotations_file=self.data_config.train_csv,
            max_seq_length=self.model_config.max_seq_length,
            stride=self.model_config.stride,
            front_view=self.data_config.use_front_view,
            side_view=self.data_config.use_side_view,
            csv_delimiter=self.data_config.csv_delimiter,
            text_column=self.data_config.text_column,
            chunk_size=self.data_config.chunk_size
        )
        
        self.val_dataset = SignLanguageDataset(
            landmarks_dir=self.data_config.data_dir,
            annotations_file=self.data_config.val_csv,
            max_seq_length=self.model_config.max_seq_length,
            stride=self.model_config.stride,
            front_view=self.data_config.use_front_view,
            side_view=self.data_config.use_side_view,
            csv_delimiter=self.data_config.csv_delimiter,
            text_column=self.data_config.text_column,
            chunk_size=self.data_config.chunk_size
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),  # Use multiple workers
            pin_memory=True,  # Pin memory for faster GPU transfer
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2,  # Prefetch next batches
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn
        ) 
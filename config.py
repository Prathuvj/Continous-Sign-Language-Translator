from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for the sign language translation model"""
    
    # Data paths
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    
    # Data processing
    max_seq_length: int = 250
    stride: int = 125
    batch_size: int = 32  # Reduced for initial loading
    num_workers: int = 4  # Adjusted for stability
    
    # Model architecture
    landmark_dim: int = 3
    conv_channels: List[int] = (64, 128, 256, 512)
    conv_kernel_size: tuple = (3, 3, 3)
    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.3
    
    # Transformer parameters
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    transformer_dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    
    # Scheduled sampling
    scheduled_sampling_initial_rate: float = 1.0
    scheduled_sampling_min_rate: float = 0.1
    scheduled_sampling_decay_rate: float = 0.01
    scheduled_sampling_decay_steps: int = 25000
    
    # Tokenizer parameters
    vocab_size: int = 30000
    min_word_freq: int = 2
    
    # Training loop
    num_epochs: int = 50
    validate_every: int = 1
    save_every: int = 2
    
    # Performance optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    
    # Multi-GPU training
    distributed_training: bool = False
    
    # Memory optimization
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # CUDA optimization
    cuda_benchmark: bool = True
    cuda_deterministic: bool = False
    cuda_memory_fraction: float = 0.8  # Use 80% of available VRAM
    
    # Logging
    use_wandb: bool = False
    
    def __post_init__(self):
        """Convert paths to Path objects and set CUDA options"""
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set CUDA options
        import torch
        if torch.cuda.is_available():
            # Force CUDA initialization
            torch.cuda.init()
            torch.cuda.set_device(0)
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.cuda_memory_fraction)
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set benchmark mode for optimized performance
            torch.backends.cudnn.benchmark = self.cuda_benchmark
            torch.backends.cudnn.deterministic = self.cuda_deterministic
            torch.backends.cudnn.enabled = True
            
            # Clear cache
            torch.cuda.empty_cache()

@dataclass
class DataConfig:
    """Configuration for data processing"""
    
    # File paths
    data_dir: Path = Path("data")  # Base data directory
    train_csv: str = "data/consolidated/how2sign_realigned_train.csv"
    val_csv: str = "data/consolidated/how2sign_realigned_val.csv"
    front_landmarks: str = "data/consolidated/normalized_landmarks_front.csv"
    side_landmarks: str = "data/consolidated/normalized_landmarks_side.csv"
    
    # Data processing
    use_front_view: bool = True
    use_side_view: bool = True
    normalize_landmarks: bool = True
    chunk_size: int = 1000  # Number of rows to load at once for memory efficiency
    
    # CSV format
    csv_delimiter: str = '\t'  # Tab-separated files
    text_column: str = 'SENTENCE'  # Column containing the text data
    
    # Augmentation
    use_augmentation: bool = True
    temporal_augmentation_factor: float = 0.2
    spatial_augmentation_factor: float = 0.1
    dropout_probability: float = 0.1
    
    def __post_init__(self):
        """Validate paths and create directories if needed"""
        # Convert paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.train_csv = Path(self.train_csv)
        self.val_csv = Path(self.val_csv)
        self.front_landmarks = Path(self.front_landmarks)
        self.side_landmarks = Path(self.side_landmarks)
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if files exist
        required_files = {
            'Training CSV': self.train_csv,
            'Validation CSV': self.val_csv,
            'Front landmarks': self.front_landmarks,
            'Side landmarks': self.side_landmarks
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(f"{name} at {path}")
        
        if missing_files:
            raise FileNotFoundError(
                "Missing required files:\n" + 
                "\n".join(missing_files)
            )

# Create default configurations
model_config = ModelConfig()
data_config = DataConfig() 
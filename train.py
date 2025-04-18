import torch
import pandas as pd
import logging
import time
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime, timedelta
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add CUDA debugging information
logging.info("PyTorch CUDA Debug Information:")
logging.info(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("CUDA is not available - training will be extremely slow on CPU!")

from model.architecture import SignLanguageTranslator
from model.trainer import SignLanguageTrainer
from model.data import SignLanguageDataModule
from model.tokenizer import SignLanguageTokenizer
from model.config import ModelConfig, DataConfig

class ProgressManager:
    """Manages progress visualization and time estimation"""
    def __init__(self):
        self.start_time = time.time()
        self.module_progress = {}
        
    def start_module(self, name: str, total_steps: int):
        """Start tracking a new module"""
        self.module_progress[name] = {
            'progress': tqdm(
                total=total_steps,
                desc=f"{name:25}",
                bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ),
            'start_time': time.time()
        }
        
    def update_module(self, name: str, steps: int = 1):
        """Update module progress"""
        if name in self.module_progress:
            self.module_progress[name]['progress'].update(steps)
            
    def finish_module(self, name: str):
        """Finish a module and show completion time"""
        if name in self.module_progress:
            progress = self.module_progress[name]['progress']
            elapsed = time.time() - self.module_progress[name]['start_time']
            progress.close()
            logging.info(f"{name} completed in {timedelta(seconds=int(elapsed))}")
            
    def estimate_total_time(self, num_epochs: int, samples_per_epoch: int, batch_size: int):
        """Estimate total training time"""
        steps_per_epoch = samples_per_epoch // batch_size
        total_steps = steps_per_epoch * num_epochs
        return total_steps, steps_per_epoch

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def setup_cuda_optimizations():
    """Set up CUDA optimizations for maximum performance"""
    if torch.cuda.is_available():
        # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cudnn.enabled = True
        
        # Set memory allocator for better memory management
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM
        torch.cuda.empty_cache()
        
        # Print CUDA information
        device_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logging.info(f"Using GPU: {device_name}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        logging.info(f"Current device: {torch.cuda.current_device()}")
        logging.info(f"Device count: {torch.cuda.device_count()}")
        logging.info(f"Initial GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    else:
        logging.warning("CUDA is not available. Training will be done on CPU!")

def build_tokenizer(config: ModelConfig, data_config: DataConfig, progress: ProgressManager) -> SignLanguageTokenizer:
    """Build and train tokenizer"""
    logging.info("Building tokenizer...")
    
    try:
        # Start tokenizer module
        progress.start_module("Tokenizer Building", 4)
        
        # Load training data
        train_df = pd.read_csv(
            data_config.train_csv,
            delimiter='\t',
            encoding='utf-8',
            on_bad_lines='skip'
        )
        progress.update_module("Tokenizer Building")
        
        if 'SENTENCE' not in train_df.columns:
            logging.error(f"'SENTENCE' column not found in CSV. Available columns: {train_df.columns.tolist()}")
            raise ValueError("Missing 'SENTENCE' column in training data")
            
        texts = train_df['SENTENCE'].tolist()
        logging.info(f"Loaded {len(texts)} training examples")
        progress.update_module("Tokenizer Building")
        
        # Initialize and build tokenizer
        tokenizer = SignLanguageTokenizer(
            vocab_size=config.vocab_size,
            min_freq=config.min_word_freq
        )
        tokenizer.build_vocab(texts)
        progress.update_module("Tokenizer Building")
        
        # Save tokenizer
        tokenizer_path = config.checkpoint_dir / 'tokenizer.json'
        tokenizer.save(str(tokenizer_path))
        progress.update_module("Tokenizer Building")
        
        progress.finish_module("Tokenizer Building")
        return tokenizer
        
    except Exception as e:
        logging.error(f"Error in tokenizer building: {str(e)}")
        raise

def main():
    """Main training function"""
    total_start_time = time.time()
    progress = ProgressManager()
    
    # Initialize progress tracking
    progress.start_module("Setup", 4)
    
    # Set up logging
    setup_logging()
    logging.info("\n" + "="*50 + "\nStarting Sign Language Translation Training\n" + "="*50)
    
    # CUDA setup and device initialization
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training requires a GPU to run efficiently.")
    
    # Force CUDA initialization
    torch.cuda.init()
    device = torch.device('cuda')
    torch.cuda.set_device(0)  # Explicitly set to first GPU
    
    # Print CUDA information
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA Version: {torch.version.cuda}")
    logging.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Current device: {torch.cuda.current_device()}")
    logging.info(f"Device count: {torch.cuda.device_count()}")
    
    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    progress.update_module("Setup")
    
    # Set up CUDA optimizations
    setup_cuda_optimizations()
    logging.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
    progress.update_module("Setup")
    
    # Initialize wandb if enabled
    if model_config.use_wandb:
        try:
            import wandb
            wandb.init(
                project="sign-language-translation",
                config={
                    **model_config.__dict__,
                    **data_config.__dict__
                }
            )
        except ImportError:
            logging.warning("wandb not installed. Continuing without wandb logging.")
            model_config.use_wandb = False
    progress.update_module("Setup")
    
    # Build tokenizer
    tokenizer = build_tokenizer(model_config, data_config, progress)
    progress.update_module("Setup")
    progress.finish_module("Setup")
    
    # Initialize data module with optimized settings
    progress.start_module("Data Loading", 3)
    data_module = SignLanguageDataModule(
        data_config=data_config,
        model_config=model_config
    )
    progress.update_module("Data Loading")
    
    # Set up datasets with optimized DataLoader settings
    try:
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        num_train_samples = len(data_module.train_dataset)
        num_val_samples = len(data_module.val_dataset)
        logging.info(f"Training samples: {num_train_samples}")
        logging.info(f"Validation samples: {num_val_samples}")
        progress.update_module("Data Loading")
    except Exception as e:
        logging.error(f"Error setting up datasets: {str(e)}")
        raise
        
    # Calculate total steps and estimate time
    total_steps, steps_per_epoch = progress.estimate_total_time(
        model_config.num_epochs,
        num_train_samples,
        model_config.batch_size
    )
    
    estimated_time_per_step = 0.5  # seconds (rough estimate)
    total_estimated_hours = (total_steps * estimated_time_per_step) / 3600
    logging.info(f"Estimated total training time: {total_estimated_hours:.1f} hours")
    progress.update_module("Data Loading")
    progress.finish_module("Data Loading")
    
    # Initialize model with optimizations
    progress.start_module("Model Setup", 2)
    model = SignLanguageTranslator(
        vocab_size=tokenizer.vocab_size_actual,
        landmark_dim=model_config.landmark_dim,
        conv_channels=model_config.conv_channels,
        lstm_hidden_dim=model_config.lstm_hidden_dim,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_decoder_layers=model_config.num_decoder_layers,
        dropout=model_config.transformer_dropout
    ).to(device)
    
    # Enable torch.compile for faster training
    if model_config.use_torch_compile:
        model = torch.compile(model)
    
    progress.update_module("Model Setup")
    
    # Initialize trainer with optimizations
    trainer = SignLanguageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=model_config.learning_rate,
        max_grad_norm=model_config.max_grad_norm,
        checkpoint_dir=str(model_config.checkpoint_dir),
        use_wandb=model_config.use_wandb,
        use_mixed_precision=model_config.use_mixed_precision,
        use_flash_attention=model_config.use_flash_attention
    )
    progress.update_module("Model Setup")
    progress.finish_module("Model Setup")
    
    # Train model with optimizations
    logging.info("\n" + "="*50 + "\nStarting Training\n" + "="*50)
    progress.start_module("Training", model_config.num_epochs)
    
    try:
        # Initialize mixed precision scaler
        scaler = amp.GradScaler('cuda', enabled=model_config.use_mixed_precision)
        
        for epoch in range(model_config.num_epochs):
            epoch_start = time.time()
            
            # Training progress bar
            train_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch+1}/{model_config.num_epochs}",
                bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [ETA: {remaining}]'
            )
            
            # Train epoch with mixed precision
            trainer.train_epoch(
                epoch, 
                tokenizer=tokenizer, 
                progress_bar=train_pbar,
                scaler=scaler
            )
            train_pbar.close()
            
            # Update overall progress
            progress.update_module("Training")
            
            # Log epoch time and memory usage
            epoch_time = time.time() - epoch_start
            remaining_epochs = model_config.num_epochs - (epoch + 1)
            estimated_remaining = timedelta(seconds=int(epoch_time * remaining_epochs))
            
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            logging.info(f"Epoch {epoch+1} completed in {timedelta(seconds=int(epoch_time))}")
            logging.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            logging.info(f"Estimated time remaining: {estimated_remaining}")
            
            # Clear cache between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    
    progress.finish_module("Training")
    
    # Save final model
    progress.start_module("Saving Model", 1)
    final_checkpoint = model_config.checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': model_config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if model_config.use_mixed_precision else None,
        'config': {
            'model': model_config.__dict__,
            'data': data_config.__dict__
        }
    }, final_checkpoint)
    progress.update_module("Saving Model")
    progress.finish_module("Saving Model")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_start_time
    logging.info(f"\nTotal training time: {timedelta(seconds=int(total_time))}")
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main() 
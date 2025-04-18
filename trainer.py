import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from typing import Dict, List, Optional, Tuple
import random
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from .metrics import MetricsCalculator, BatchMetricsTracker, calculate_batch_metrics
import time

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class ScheduledSampling:
    """Implements scheduled sampling with exponential decay"""
    def __init__(
        self,
        initial_rate: float = 1.0,
        min_rate: float = 0.1,
        decay_rate: float = 0.01,
        decay_steps: int = 1000
    ):
        """
        Args:
            initial_rate: Initial teacher forcing rate
            min_rate: Minimum teacher forcing rate
            decay_rate: Rate of exponential decay
            decay_steps: Number of steps for decay
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
    def get_rate(self, step: int) -> float:
        """Get teacher forcing rate for current step"""
        rate = self.initial_rate * math.exp(-self.decay_rate * step / self.decay_steps)
        return max(self.min_rate, rate)

class SignLanguageTrainer:
    """Trainer for sign language translation model"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        use_mixed_precision: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Args:
            model: The sign language translation model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            use_mixed_precision: Whether to use mixed precision training
            use_flash_attention: Whether to use flash attention for faster training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training optimizations
        self.use_mixed_precision = use_mixed_precision
        self.use_flash_attention = use_flash_attention
        
        # Check wandb availability
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and not WANDB_AVAILABLE:
            logging.warning("WandB was requested but is not installed. Continuing without WandB logging.")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function (ignore padding index)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding index
        
        # Scheduled sampling
        self.scheduled_sampling = ScheduledSampling()
        
        # Initialize step counter
        self.global_step = 0
        
        # Add metrics calculator
        self.metrics_calculator = MetricsCalculator()
        self.batch_tracker = BatchMetricsTracker()
        
    def _decode_predictions(self, logits: torch.Tensor, tokenizer) -> List[str]:
        """Convert model outputs to text"""
        predictions = logits.argmax(dim=-1)
        return [tokenizer.decode(pred) for pred in predictions]
        
    def _teacher_forcing_step(
        self,
        front_seq: torch.Tensor,
        side_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        target_seq: torch.Tensor,
        target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform a training step with teacher forcing"""
        # Start batch tracking
        self.batch_tracker.start_batch(
            batch_size=front_seq.size(0),
            num_frames=front_seq.size(1)
        )
        
        # Forward pass
        output = self.model(
            front_sequences=front_seq,
            side_sequences=side_seq,
            seq_lengths=seq_lengths,
            target_sequences=target_seq[:, :-1],
            target_padding_mask=target_mask[:, :-1]
        )
        
        # End batch tracking
        performance_metrics = self.batch_tracker.end_batch()
        
        # Compute loss
        logits = output['logits']
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            target_seq[:, 1:].contiguous().view(-1)
        )
        
        return loss, {'logits': logits, **performance_metrics}
        
    def _scheduled_sampling_step(
        self,
        front_seq: torch.Tensor,
        side_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        target_seq: torch.Tensor,
        target_mask: torch.Tensor,
        teacher_forcing_rate: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform a training step with scheduled sampling"""
        # Start batch tracking
        self.batch_tracker.start_batch(
            batch_size=front_seq.size(0),
            num_frames=front_seq.size(1)
        )
        
        batch_size, max_len = target_seq.size()
        vocab_size = self.model.decoder.output_layer.out_features
        
        # Initialize decoder input with BOS token
        decoder_input = target_seq[:, 0].unsqueeze(1)  # BOS token
        total_loss = 0
        
        # Generate sequence
        for t in range(1, max_len):
            # Forward pass
            output = self.model(
                front_sequences=front_seq,
                side_sequences=side_seq,
                seq_lengths=seq_lengths,
                target_sequences=decoder_input,
                target_padding_mask=target_mask[:, :t]
            )
            
            logits = output['logits']
            
            # Compute loss for current step
            step_loss = self.criterion(
                logits[:, -1],
                target_seq[:, t]
            )
            total_loss += step_loss
            
            # Scheduled sampling
            if random.random() < teacher_forcing_rate:
                # Teacher forcing: use ground truth
                next_token = target_seq[:, t].unsqueeze(1)
            else:
                # Use model prediction
                next_token = logits[:, -1].argmax(dim=-1).unsqueeze(1)
            
            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # End batch tracking
        performance_metrics = self.batch_tracker.end_batch()
        
        return total_loss / (max_len - 1), {'logits': logits, **performance_metrics}
    
    def train_epoch(self, epoch: int, tokenizer, progress_bar=None, scaler=None) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Debug GPU usage
        if torch.cuda.is_available():
            logging.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"Model device: {next(self.model.parameters()).device}")
        else:
            logging.warning("Training on CPU - this will be very slow!")
        
        total_metrics = {
            'loss': 0.0,
            'wer': 0.0,
            'bleu': 0.0,
            'inference_time': 0.0,
            'frames_per_second': 0.0,
            'memory_used': 0.0,
            'gpu_utilization': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Debug first batch
            if batch_idx == 0:
                logging.info("First batch device check:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logging.info(f"{k} device: {v.device}")
            
            # Get teacher forcing rate for current step
            teacher_forcing_rate = self.scheduled_sampling.get_rate(self.global_step)
            
            # Explicitly move batch to device
            front_seq = batch['front_sequences'].to(self.device, non_blocking=True) if batch['front_sequences'] is not None else None
            side_seq = batch['side_sequences'].to(self.device, non_blocking=True) if batch['side_sequences'] is not None else None
            seq_lengths = batch['seq_lengths'].to(self.device, non_blocking=True)
            target_seq = batch['target_sequences'].to(self.device, non_blocking=True)
            target_mask = batch['target_padding_mask'].to(self.device, non_blocking=True)
            
            # Debug GPU memory after moving batch
            if batch_idx == 0 and torch.cuda.is_available():
                logging.info(f"GPU Memory after first batch transfer: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, outputs = self._scheduled_sampling_step(
                        front_seq=front_seq,
                        side_seq=side_seq,
                        seq_lengths=seq_lengths,
                        target_seq=target_seq,
                        target_mask=target_mask,
                        teacher_forcing_rate=teacher_forcing_rate
                    )
                
                # Debug GPU memory after forward pass
                if batch_idx == 0 and torch.cuda.is_available():
                    logging.info(f"GPU Memory after first forward pass: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Clip gradients
                if self.max_grad_norm > 0:
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights with gradient scaling
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Regular forward pass without mixed precision
                loss, outputs = self._scheduled_sampling_step(
                    front_seq=front_seq,
                    side_seq=side_seq,
                    seq_lengths=seq_lengths,
                    target_seq=target_seq,
                    target_mask=target_mask,
                    teacher_forcing_rate=teacher_forcing_rate
                )
                
                # Regular backward pass
                loss.backward()
                
                # Clip gradients
                if self.max_grad_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                self.optimizer.step()
            
            # Debug GPU memory after backward pass
            if batch_idx == 0 and torch.cuda.is_available():
                logging.info(f"GPU Memory after first backward pass: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            # Calculate batch processing time
            batch_time = time.time() - batch_start
            
            # Profile GPU utilization if available
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                memory_utilization = current_memory / torch.cuda.get_device_properties(0).total_memory
                total_metrics['gpu_utilization'] += memory_utilization
            
            # Increment global step
            self.global_step += 1
            
            # Calculate batch metrics
            batch_metrics = calculate_batch_metrics(
                outputs['logits'],
                target_seq,
                self._decode_predictions(outputs['logits'], tokenizer),
                [tokenizer.decode(seq) for seq in target_seq],
                outputs
            )
            
            # Update total metrics
            for key, value in batch_metrics.items():
                total_metrics[key] += value
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    'train/step': self.global_step,
                    'train/loss': loss.item(),
                    'train/batch_time': batch_time,
                    'train/memory_utilization': memory_utilization if torch.cuda.is_available() else 0,
                    **{f'train/{k}': v for k, v in batch_metrics.items()}
                })
            
            # Update progress bar if provided
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'wer': f"{batch_metrics['wer']:.2f}",
                    'bleu': f"{batch_metrics['bleu']:.2f}",
                    'time/batch': f"{batch_time:.3f}s"
                })
            
            # Clear cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        # Log final memory stats
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            logging.info(f"Epoch {epoch} memory usage:")
            logging.info(f"  Peak memory: {peak_memory / 1024**3:.2f}GB")
            logging.info(f"  Final memory: {end_memory / 1024**3:.2f}GB")
            logging.info(f"  Memory increase: {(end_memory - start_memory) / 1024**3:.2f}GB")
        
        return total_metrics
    
    @torch.no_grad()
    def validate(self, tokenizer) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_metrics = {
            'val_loss': 0.0,
            'val_wer': 0.0,
            'val_bleu': 0.0,
            'val_inference_time': 0.0,
            'val_frames_per_second': 0.0,
            'val_memory_used': 0.0
        }
        num_batches = len(self.val_loader)
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            front_seq = batch['front_sequences'].to(self.device)
            side_seq = batch['side_sequences'].to(self.device) if batch['side_sequences'] is not None else None
            seq_lengths = batch['seq_lengths'].to(self.device)
            target_seq = batch['texts'].to(self.device)
            target_mask = (target_seq != 0).float().to(self.device)
            
            # Forward pass (always use teacher forcing during validation)
            loss, step_outputs = self._teacher_forcing_step(
                front_seq, side_seq, seq_lengths, target_seq, target_mask
            )
            
            # Calculate metrics
            predictions = self._decode_predictions(step_outputs['logits'], tokenizer)
            targets = [tokenizer.decode(t) for t in target_seq]
            
            batch_metrics = calculate_batch_metrics(
                predictions=predictions,
                targets=targets,
                performance_metrics={
                    k: v for k, v in step_outputs.items()
                    if k != 'logits'
                }
            )
            
            # Update total metrics
            total_metrics['val_loss'] += loss.item()
            total_metrics['val_wer'] += batch_metrics['wer']
            total_metrics['val_bleu'] += batch_metrics['bleu']
            total_metrics['val_inference_time'] += batch_metrics['inference_time']
            total_metrics['val_frames_per_second'] += batch_metrics['frames_per_second']
            total_metrics['val_memory_used'] += batch_metrics['memory_used']
        
        # Calculate averages
        for k in total_metrics:
            total_metrics[k] /= num_batches
            
        return total_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save a checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if self.use_wandb:
            wandb.save(str(checkpoint_path))
    
    def train(
        self,
        num_epochs: int,
        validate_every: int = 1,
        save_every: int = 1
    ):
        """Train the model"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % validate_every == 0:
                val_metrics = self.validate()
                train_metrics.update(val_metrics)
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, train_metrics)
            
            # Regular checkpoint saving
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_metrics)
            
            # Log metrics
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics
                })
            
            print(f"Epoch {epoch} metrics:", train_metrics) 
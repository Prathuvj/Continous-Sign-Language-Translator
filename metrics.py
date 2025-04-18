import torch
import numpy as np
from typing import List, Dict, Union
from sacrebleu.metrics import BLEU
import editdistance
import time
from dataclasses import dataclass
from torchmetrics.text import WordErrorRate, BLEUScore
from collections import deque
import logging

@dataclass
class PerformanceMetrics:
    """Container for real-time performance metrics"""
    inference_time: float  # Time in milliseconds
    frames_per_second: float
    memory_used: float  # Memory in MB
    batch_size: int

class MetricsCalculator:
    """Calculator for translation and performance metrics"""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Size of sliding window for performance metrics
        """
        # Initialize metric computers
        self.wer_metric = WordErrorRate()
        self.bleu_metric = BLEUScore()
        
        # Performance tracking
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.fps_values = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        
    def calculate_wer(self, predictions: List[str], targets: List[str]) -> float:
        """
        Calculate Word Error Rate
        
        Args:
            predictions: List of predicted sentences
            targets: List of target sentences
        Returns:
            Word Error Rate score
        """
        try:
            return self.wer_metric(predictions, targets).item()
        except Exception as e:
            logging.error(f"Error calculating WER: {e}")
            return float('nan')
    
    def calculate_bleu(self, predictions: List[str], targets: List[str]) -> float:
        """
        Calculate BLEU score
        
        Args:
            predictions: List of predicted sentences
            targets: List of target sentences
        Returns:
            BLEU score
        """
        try:
            return self.bleu_metric(predictions, [[t] for t in targets]).item()
        except Exception as e:
            logging.error(f"Error calculating BLEU: {e}")
            return float('nan')
    
    def update_performance_metrics(
        self,
        batch_size: int,
        inference_time: float,
        num_frames: int,
        memory_used: float
    ) -> None:
        """
        Update performance metrics
        
        Args:
            batch_size: Size of the processed batch
            inference_time: Time taken for inference (in seconds)
            num_frames: Number of frames processed
            memory_used: Memory used in MB
        """
        self.inference_times.append(inference_time * 1000)  # Convert to ms
        self.fps_values.append(num_frames / inference_time)
        self.memory_usage.append(memory_used)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get average performance metrics over the sliding window"""
        if not self.inference_times:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0)
            
        return PerformanceMetrics(
            inference_time=np.mean(self.inference_times),
            frames_per_second=np.mean(self.fps_values),
            memory_used=np.mean(self.memory_usage),
            batch_size=len(self.inference_times)
        )
    
    def reset_performance_metrics(self) -> None:
        """Reset all performance metric trackers"""
        self.inference_times.clear()
        self.fps_values.clear()
        self.memory_usage.clear()

class BatchMetricsTracker:
    """Tracks metrics for a single batch during training/validation"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.batch_size = 0
        self.num_frames = 0
        self.memory_start = 0
        
    def start_batch(self, batch_size: int, num_frames: int):
        """Start tracking a new batch"""
        self.start_time = time.perf_counter()
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.memory_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
    def end_batch(self) -> Dict[str, float]:
        """End batch tracking and return metrics"""
        self.end_time = time.perf_counter()
        current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        inference_time = self.end_time - self.start_time
        memory_used = (current_memory - self.memory_start) / (1024 * 1024)  # Convert to MB
        
        return {
            'inference_time': inference_time,
            'frames_per_second': self.num_frames / inference_time,
            'memory_used': memory_used,
            'batch_size': self.batch_size
        }

def postprocess_text(text: str) -> str:
    """
    Postprocess text for metric calculation
    
    Args:
        text: Input text to process
    Returns:
        Processed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove punctuation (can be customized based on needs)
    text = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    
    return text

def calculate_batch_metrics(
    predictions: List[str],
    targets: List[str],
    performance_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate all metrics for a batch
    
    Args:
        predictions: List of predicted sentences
        targets: List of target sentences
        performance_metrics: Dictionary of performance metrics
    Returns:
        Dictionary containing all metrics
    """
    # Process texts
    processed_preds = [postprocess_text(p) for p in predictions]
    processed_targets = [postprocess_text(t) for t in targets]
    
    # Initialize metrics calculator
    calculator = MetricsCalculator()
    
    # Calculate translation metrics
    wer = calculator.calculate_wer(processed_preds, processed_targets)
    bleu = calculator.calculate_bleu(processed_preds, processed_targets)
    
    # Combine with performance metrics
    return {
        'wer': wer,
        'bleu': bleu,
        **performance_metrics
    } 
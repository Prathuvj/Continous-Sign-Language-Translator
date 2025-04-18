import torch
from typing import List, Dict, Optional
from collections import Counter
import re
from pathlib import Path
import json
import pandas as pd
import numpy as np

class SignLanguageTokenizer:
    """Tokenizer for sign language translation text"""
    
    # Special tokens
    PAD_token = "[PAD]"
    SOS_token = "[SOS]"  # Start of sentence
    EOS_token = "[EOS]"  # End of sentence
    UNK_token = "[UNK]"  # Unknown token
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_freq: int = 2,
        lowercase: bool = True
    ):
        """
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a word to be included
            lowercase: Whether to convert text to lowercase
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.lowercase = lowercase
        
        # Initialize vocabulary
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.PAD_token, self.SOS_token, self.EOS_token, self.UNK_token]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization"""
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic punctuation normalization
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            text = self._preprocess_text(text)
            words = text.split()
            self.word_freq.update(words)
        
        # Filter by minimum frequency and vocab size
        vocab_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ][:self.vocab_size - len(self.word2idx)]
        
        # Add words to vocabulary
        for word in vocab_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode text to tensor of indices
        
        Args:
            text: Input text
            max_length: Maximum sequence length (including SOS/EOS)
        Returns:
            Tensor of indices
        """
        text = self._preprocess_text(text)
        words = text.split()
        
        # Add SOS and EOS
        tokens = [self.SOS_token] + words + [self.EOS_token]
        
        # Truncate if needed
        if max_length is not None:
            tokens = tokens[:max_length-1] + [self.EOS_token]
        
        # Convert to indices
        indices = [
            self.word2idx.get(token, self.word2idx[self.UNK_token])
            for token in tokens
        ]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode indices to text
        
        Args:
            indices: Tensor of indices
        Returns:
            Decoded text
        """
        tokens = [self.idx2word.get(idx.item(), self.UNK_token) for idx in indices]
        
        # Remove special tokens
        tokens = [
            token for token in tokens
            if token not in {self.PAD_token, self.SOS_token, self.EOS_token}
        ]
        
        return ' '.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer vocabulary and configuration"""
        save_dict = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'lowercase': self.lowercase,
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'word_freq': dict(self.word_freq)
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SignLanguageTokenizer':
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            save_dict = json.load(f)
        
        tokenizer = cls(
            vocab_size=save_dict['vocab_size'],
            min_freq=save_dict['min_freq'],
            lowercase=save_dict['lowercase']
        )
        
        tokenizer.word2idx = save_dict['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in save_dict['idx2word'].items()}
        tokenizer.word_freq = Counter(save_dict['word_freq'])
        
        return tokenizer
    
    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size"""
        return len(self.word2idx)
    
    def get_pad_idx(self) -> int:
        """Get padding token index"""
        return self.word2idx[self.PAD_token]
    
    def get_sos_idx(self) -> int:
        """Get start of sentence token index"""
        return self.word2idx[self.SOS_token]
    
    def get_eos_idx(self) -> int:
        """Get end of sentence token index"""
        return self.word2idx[self.EOS_token] 
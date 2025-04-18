import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Optional, Tuple

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on relevant body parts"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_landmarks, features)
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, seq_len, num_landmarks, _ = x.size()
        
        # Reshape for attention computation
        x_reshaped = x.view(batch_size * seq_len, num_landmarks, -1)
        
        # Compute attention weights
        attention_weights = self.attention(x_reshaped)  # (batch_size * seq_len, num_landmarks, 1)
        
        # Apply attention
        attended = torch.bmm(
            attention_weights.transpose(1, 2),
            x_reshaped
        )  # (batch_size * seq_len, 1, features)
        
        # Reshape back
        attended = attended.view(batch_size, seq_len, -1)
        attention_weights = attention_weights.view(batch_size, seq_len, num_landmarks)
        
        return attended, attention_weights

class Conv3DBlock(nn.Module):
    """3D Convolutional block with batch normalization and ReLU"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0)
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class TemporalEncoder(nn.Module):
    """Temporal encoder with LSTM and attention"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            lengths: Sequence lengths
        Returns:
            Tuple of (outputs, (hidden_state, cell_state))
        """
        # Pack sequence for efficient computation
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through LSTM
        outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        return self.dropout(outputs), (hidden, cell)

class TransformerDecoder(nn.Module):
    """Transformer decoder for text generation"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence
            memory: Output from encoder
            tgt_mask: Target sequence mask
            tgt_padding_mask: Target padding mask
            memory_key_padding_mask: Memory key padding mask
        """
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return self.output_layer(output)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class SignLanguageTranslator(nn.Module):
    """Complete sign language translation model"""
    def __init__(
        self,
        vocab_size: int,
        landmark_dim: int = 3,
        conv_channels: List[int] = [64, 128, 256],
        lstm_hidden_dim: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 3D CNN for spatial feature extraction
        self.conv_layers = nn.ModuleList([
            Conv3DBlock(
                in_channels=1 if i == 0 else conv_channels[i-1],
                out_channels=conv_channels[i],
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            )
            for i in range(len(conv_channels))
        ])
        
        # Spatial attention
        conv_output_dim = conv_channels[-1] * landmark_dim
        self.spatial_attention = SpatialAttention(
            input_dim=conv_output_dim,
            hidden_dim=lstm_hidden_dim
        )
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=conv_output_dim,
            hidden_dim=lstm_hidden_dim,
            dropout=dropout
        )
        
        # Linear projection to match transformer dimensions
        self.projection = nn.Linear(
            lstm_hidden_dim * 2,  # * 2 for bidirectional
            d_model
        )
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
    def forward(
        self,
        front_sequences: torch.Tensor,
        side_sequences: torch.Tensor,
        seq_lengths: torch.Tensor,
        target_sequences: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            front_sequences: Front view sequences (batch_size, seq_len, num_landmarks, features)
            side_sequences: Side view sequences
            seq_lengths: Sequence lengths
            target_sequences: Target text sequences
            target_padding_mask: Target padding mask
        """
        batch_size = front_sequences.size(0)
        
        # Process front view
        front_features = self._process_view(front_sequences, seq_lengths)
        
        # Process side view if available
        if side_sequences is not None:
            side_features = self._process_view(side_sequences, seq_lengths)
            # Combine views
            features = torch.cat([front_features, side_features], dim=-1)
        else:
            features = front_features
        
        # Project to transformer dimensions
        memory = self.projection(features)
        
        # Generate target mask for training
        if target_sequences is not None:
            tgt_mask = self._generate_square_subsequent_mask(
                target_sequences.size(1)
            ).to(target_sequences.device)
        else:
            tgt_mask = None
        
        # Decode
        output = self.decoder(
            target_sequences,
            memory,
            tgt_mask=tgt_mask,
            tgt_padding_mask=target_padding_mask,
            memory_key_padding_mask=None  # All memory tokens are valid
        )
        
        return {
            'logits': output,
            'encoder_output': memory
        }
    
    def _process_view(
        self,
        sequences: torch.Tensor,
        seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Process a single view through CNN and LSTM"""
        batch_size, seq_len, num_landmarks, features = sequences.size()
        
        # Reshape for 3D CNN
        x = sequences.view(batch_size, 1, seq_len, num_landmarks, features)
        
        # Apply 3D CNN layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape back
        x = x.view(batch_size, seq_len, num_landmarks, -1)
        
        # Apply spatial attention
        attended, _ = self.spatial_attention(x)
        
        # Apply temporal encoder
        temporal_features, _ = self.temporal_encoder(attended, seq_lengths)
        
        return temporal_features
    
    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate mask for transformer decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 
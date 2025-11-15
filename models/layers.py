"""
Additional layers for the transformer architecture.
These will be implemented in Week 2.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding to give the model information about token positions.
    Uses sine and cosine functions of different frequencies.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        # TODO: Implement in Week 2
        # Create positional encoding matrix
        pass
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        # TODO: Implement in Week 2
        pass


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Implement in Week 2
        # Two linear transformations with ReLU activation
        pass
    
    def forward(self, x):
        # TODO: Implement in Week 2
        pass


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block.
    Consists of:
    1. Masked Multi-Head Self-Attention
    2. Feed-Forward Network
    Both with residual connections and layer normalization.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Implement in Week 2
        # Multi-head attention
        # Feed-forward network
        # Layer normalization
        # Dropout
        pass
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # TODO: Implement in Week 2
        # 1. Self-attention with residual connection and layer norm
        # 2. Feed-forward with residual connection and layer norm
        pass

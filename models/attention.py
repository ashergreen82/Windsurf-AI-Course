"""
Attention mechanisms for the Mini-GPT transformer.
These will be implemented in Week 2.
"""

import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention from "Attention Is All You Need"
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, n_heads, seq_len, d_k)
            key: (batch_size, n_heads, seq_len, d_k)
            value: (batch_size, n_heads, seq_len, d_v)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            output: (batch_size, n_heads, seq_len, d_v)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        # TODO: Implement in Week 2
        # Calculate attention scores
        # Apply mask if provided
        # Apply softmax
        # Apply dropout
        # Weight the values
        pass


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Allows the model to attend to information from different representation subspaces.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # TODO: Implement in Week 2
        # Create linear layers for Q, K, V projections
        # Create output projection
        pass
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # TODO: Implement in Week 2
        # 1. Linear projections in batch from d_model => h x d_k 
        # 2. Apply attention on all the projected vectors in batch
        # 3. Concat and apply final linear
        pass

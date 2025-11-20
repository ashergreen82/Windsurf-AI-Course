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
    
    This is the core mechanism that allows the model to "attend" to 
    different parts of the input sequence.
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: (batch_size, n_heads, seq_len, d_k) - What we're looking for
            key: (batch_size, n_heads, seq_len, d_k) - What we're matching against
            value: (batch_size, n_heads, seq_len, d_v) - The actual content
            mask: Optional (seq_len, seq_len) - Prevents attention to certain positions
            
        Returns:
            output: (batch_size, n_heads, seq_len, d_v) - Attention-weighted values
            attention_weights: (batch_size, n_heads, seq_len, seq_len) - Attention scores
        """
        # Get dimension for scaling (d_k)
        d_k = query.size(-1)
        
        # Step 1: Compute attention scores
        # scores = Q @ K^T
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale by sqrt(d_k) to prevent vanishing gradients
        # Without scaling, softmax can become too peaked
        scores = scores / math.sqrt(d_k)
        
        # Step 3: Apply mask if provided (for causal/autoregressive attention)
        # Set masked positions to large negative value so softmax makes them ~0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights (probabilities)
        # Each row sums to 1.0
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Step 5: Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Multiply weights by values to get weighted sum
        # This is the actual "attending" - focusing on relevant parts
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


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

"""
Main Mini-GPT transformer model.
This will be implemented in Week 2.
"""

import torch
import torch.nn as nn
from .layers import PositionalEncoding, TransformerBlock


class MiniGPT(nn.Module):
    """
    A decoder-only transformer for text generation (GPT-style).
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, max_seq_len=256, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        # TODO: Implement in Week 2
        # Token embedding
        # Positional encoding
        # Transformer blocks
        # Output projection
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) - token indices
            mask: Optional attention mask
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # TODO: Implement in Week 2
        # 1. Embed tokens
        # 2. Add positional encoding
        # 3. Pass through transformer blocks
        # 4. Project to vocabulary
        pass
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given a context.
        
        Args:
            idx: (batch_size, seq_len) - context tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k most likely tokens
            
        Returns:
            Generated sequence including context
        """
        # TODO: Implement in Week 4
        # This is for text generation
        pass

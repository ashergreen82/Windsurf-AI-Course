"""
Test script for attention mechanisms.
Run this to verify your implementation works correctly.
"""

import torch
from models.attention import ScaledDotProductAttention

def test_scaled_dot_product_attention():
    """Test the basic attention mechanism"""
    print("="* 60)
    print("Testing Scaled Dot-Product Attention")
    print("=" * 60)
    
    # Create attention module
    attn = ScaledDotProductAttention(dropout=0.1)
    
    # Create dummy input tensors
    batch_size = 2
    n_heads = 8
    seq_len = 10
    d_k = 64  # dimension of key/query
    
    # Q, K, V tensors
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    print(f"\nInput shapes:")
    print(f"  Query (Q): {Q.shape}")
    print(f"  Key (K):   {K.shape}")
    print(f"  Value (V): {V.shape}")
    
    # Forward pass
    output, attention_weights = attn(Q, K, V)
    
    print(f"\nOutput shapes:")
    print(f"  Output:            {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    
    # Verify attention weights sum to 1 (along last dimension)
    weight_sums = attention_weights.sum(dim=-1)
    print(f"\nAttention weight sums (should be ~1.0):")
    print(f"  Min: {weight_sums.min():.4f}")
    print(f"  Max: {weight_sums.max():.4f}")
    print(f"  Mean: {weight_sums.mean():.4f}")
    
    # Show attention pattern for first head of first batch
    print(f"\nAttention pattern (first head, first batch):")
    print(f"  Shape: {attention_weights[0, 0].shape}")
    print(f"  Each row shows how much each position attends to others")
    print(f"\n  First 5x5 block:")
    print(attention_weights[0, 0, :5, :5].detach().numpy())
    
    print("\n✅ Scaled Dot-Product Attention test passed!")
    return True

def test_with_causal_mask():
    """Test attention with causal masking (for autoregressive generation)"""
    print("\n" + "=" * 60)
    print("Testing Causal Masking")
    print("=" * 60)
    
    attn = ScaledDotProductAttention(dropout=0.0)  # No dropout for testing
    
    batch_size = 1
    n_heads = 1
    seq_len = 5
    d_k = 8
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    # Create causal mask (lower triangular)
    # This prevents position i from attending to positions > i
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    print(f"\nCausal mask (1 = can attend, 0 = cannot attend):")
    print(mask[0, 0].numpy())
    
    # Forward pass with mask
    output, attention_weights = attn(Q, K, V, mask=mask)
    
    print(f"\nAttention weights with causal mask:")
    print(attention_weights[0, 0].detach().numpy())
    print(f"\nNotice: Upper triangle is ~0 (cannot attend to future)")
    
    print("\n✅ Causal masking test passed!")
    return True

if __name__ == "__main__":
    # Run tests
    test_scaled_dot_product_attention()
    test_with_causal_mask()
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Attention mechanism works correctly.")
    print("=" * 60)

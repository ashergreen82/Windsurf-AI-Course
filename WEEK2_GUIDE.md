# Week 2: Core Architecture (Nov 17-24, 2024)

This week you'll implement the transformer architecture from scratch, learning how attention works.

## Learning Goals

By the end of Week 2, you'll understand:
- How scaled dot-product attention works mathematically
- Why we use multiple attention heads
- How positional encoding gives the model sequence order information
- How transformer blocks combine attention and feed-forward layers

## Week 2 Roadmap

### Day 1-2: Scaled Dot-Product Attention
**The Core Formula**: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`

**Theory:**
- Query (Q): "What am I looking for?"
- Key (K): "What information do I have?"
- Value (V): "What is the actual information?"
- Scaling by sqrt(d_k) prevents exploding gradients

**Implementation:** `models/attention.py` - ScaledDotProductAttention class

### Day 3-4: Multi-Head Attention
**Why Multiple Heads?**
- Each head can attend to different aspects
- Like having multiple "viewpoints" on the data
- Heads learn different patterns (syntax, semantics, etc.)

**Implementation:** `models/attention.py` - MultiHeadAttention class

### Day 5: Positional Encoding
**The Problem:** Attention has no concept of order (it's permutation invariant)
**The Solution:** Add position information using sine/cosine functions

**Formula:**
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

**Implementation:** `models/layers.py` - PositionalEncoding class

### Day 6: Feed-Forward & Transformer Block
**Feed-Forward Network:** Simple 2-layer MLP applied to each position
**Transformer Block:** Combines attention + FFN with residual connections

**Implementation:** `models/layers.py` - FeedForward, TransformerBlock

### Day 7: Assemble the Full Model
**Put it all together:** `models/transformer.py` - MiniGPT class

- Token embeddings
- Positional encoding
- Stack of transformer blocks
- Output projection to vocabulary

## Key Concepts to Master

### 1. Attention Mechanism
```
scores = Q @ K.T / sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights @ V
```

### 2. Masking (Causal/Autoregressive)
- Prevent attention to future tokens
- Essential for text generation
- Upper triangular mask

### 3. Residual Connections
- `output = LayerNorm(x + Sublayer(x))`
- Helps gradient flow
- Stabilizes training

### 4. Why Transformers Work
- **Parallelization**: Unlike RNNs, process all positions at once
- **Long-range dependencies**: Direct connections between any two positions
- **Scalability**: Performance improves with more data and compute

## Testing Your Implementation

After each component, test it:

```python
# Test attention
from models.attention import ScaledDotProductAttention
import torch

attn = ScaledDotProductAttention()
Q = torch.randn(2, 8, 10, 64)  # batch, heads, seq_len, d_k
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)

output, weights = attn(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

## Resources for This Week

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper (Section 3.1-3.2)
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - Lecture on Transformers

## Debugging Tips

- **Shape mismatches**: Print tensor shapes at every step
- **NaN values**: Usually from division by zero or numerical instability
- **Masking issues**: Visualize attention weights as heatmap
- **Gradient problems**: Check if gradients are flowing with `tensor.requires_grad`

## Week 2 Checklist

- [ ] Understand Q, K, V concept
- [ ] Implement scaled dot-product attention
- [ ] Implement multi-head attention
- [ ] Test attention with dummy data
- [ ] Implement positional encoding
- [ ] Implement feed-forward network
- [ ] Implement transformer block
- [ ] Assemble full MiniGPT model
- [ ] Test forward pass with real tokens

## Next Week Preview

Week 3: Training! We'll implement the training loop, loss calculation, optimization, and see our model learn to generate text.

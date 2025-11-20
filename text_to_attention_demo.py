"""
Demonstration: Full pipeline from text → attention
Shows exactly how "the" becomes numbers at each step
"""

import torch
import torch.nn as nn
from utils.tokenizer import CharTokenizer

print("=" * 70)
print("TEXT → ATTENTION: Complete Pipeline Demo")
print("=" * 70)

# ============================================================================
# STEP 1: TEXT → TOKEN IDs (Tokenization)
# ============================================================================
print("\n📝 STEP 1: Text → Token IDs (Tokenization)")
print("-" * 70)

text = "the cat sat"
print(f"Original text: '{text}'")

# Create and build tokenizer
tokenizer = CharTokenizer()
tokenizer.build_vocab(text)

print(f"\nVocabulary mapping:")
for char, idx in sorted(tokenizer.char_to_idx.items()):
    print(f"  '{char}' → {idx}")

# Encode text to IDs
token_ids = tokenizer.encode(text)
print(f"\nEncoded token IDs: {token_ids}")
print(f"Visualization:")
for char, token_id in zip(text, token_ids):
    print(f"  '{char}' → {token_id}")

# ============================================================================
# STEP 2: TOKEN IDs → EMBEDDINGS (Learned Vectors)
# ============================================================================
print("\n\n🎯 STEP 2: Token IDs → Embeddings (Learned Vectors)")
print("-" * 70)

vocab_size = tokenizer.vocab_size
d_model = 8  # Small for demo (real model uses 256)

# Create embedding layer (this is learned during training)
embedding_layer = nn.Embedding(vocab_size, d_model)

print(f"Embedding layer: {vocab_size} tokens × {d_model} dimensions")
print(f"This is a {vocab_size}×{d_model} matrix of learnable parameters\n")

# Convert token IDs to embeddings
token_tensor = torch.tensor(token_ids)
embeddings = embedding_layer(token_tensor)

print(f"Input token IDs shape: {token_tensor.shape}")
print(f"Output embeddings shape: {embeddings.shape}")
print(f"  → Each of {len(token_ids)} tokens became a {d_model}-dim vector\n")

# Show embeddings for "the"
print(f"Word 'the' consists of characters: t-h-e")
print(f"Token IDs for 'the': {token_ids[:3]}")
print(f"\nEmbeddings for 'the':")
for i, char in enumerate(['t', 'h', 'e']):
    emb = embeddings[i].detach().numpy()
    print(f"  '{char}' (ID={token_ids[i]}): {emb}")

# ============================================================================
# STEP 3: EMBEDDINGS → Q, K, V (Linear Projections)
# ============================================================================
print("\n\n🔄 STEP 3: Embeddings → Q, K, V (Linear Projections)")
print("-" * 70)

d_k = 4  # Dimension for Q, K (smaller for demo)

# These are learned transformations
W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_k, bias=False)

# Project embeddings to Q, K, V
Q = W_q(embeddings)
K = W_k(embeddings)
V = W_v(embeddings)

print(f"Q (Query) shape: {Q.shape}")
print(f"K (Key) shape: {K.shape}")
print(f"V (Value) shape: {V.shape}")

print(f"\nQ for 'the':")
for i, char in enumerate(['t', 'h', 'e']):
    q = Q[i].detach().numpy()
    print(f"  '{char}': {q}")

# ============================================================================
# STEP 4: ATTENTION COMPUTATION
# ============================================================================
print("\n\n⚡ STEP 4: Attention Computation")
print("-" * 70)

# Compute attention scores (dot products)
scores = torch.matmul(Q, K.transpose(-1, -0))
print(f"Attention scores shape: {scores.shape}")
print(f"  → {len(token_ids)} × {len(token_ids)} matrix")
print(f"  → scores[i,j] = how much position i attends to position j\n")

# Scale
import math
scores = scores / math.sqrt(d_k)

# Softmax to get attention weights
attention_weights = torch.softmax(scores, dim=-1)

print(f"Attention weights (after softmax):")
print(f"  Each row sums to 1.0 (probability distribution)\n")

# Show attention for first 3 positions (t, h, e)
chars = list(text)
print(f"How much does each character attend to others?")
print(f"Rows = from, Columns = to\n")
print("     ", end="")
for char in chars[:7]:
    print(f"'{char}'   ", end="")
print()

for i in range(min(3, len(chars))):
    print(f"'{chars[i]}': ", end="")
    for j in range(min(7, len(chars))):
        print(f"{attention_weights[i,j]:.2f}  ", end="")
    print()

print(f"\nExample: Character 't' attends to:")
for j, char in enumerate(chars[:7]):
    weight = attention_weights[0, j].item()
    print(f"  '{char}': {weight:.1%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("📊 SUMMARY: What Happens to 'the'")
print("=" * 70)
print("""
1. TEXT: "the"
   ↓
2. TOKENIZE: ['t', 'h', 'e'] → [token_id_t, token_id_h, token_id_e]
   ↓
3. EMBED: Each ID → learned 256-dim vector (floats, not binary!)
   ↓
4. PROJECT: Apply W_q, W_k, W_v transformations
   ↓
5. ATTENTION: Compute which positions attend to which
   ↓
6. OUTPUT: Context-aware representation

Key Points:
- NO ASCII conversion!
- NO binary!
- Just: text → IDs → learned vectors → attention
- All vectors are LEARNED during training (not hardcoded)
""")

print("=" * 70)
print("\n✅ Run this script to see the complete pipeline!")
print("Command: python text_to_attention_demo.py")

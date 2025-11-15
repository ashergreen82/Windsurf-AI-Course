# Week 1: Data & Tokenization (Nov 12-19, 2024)

This week focuses on understanding how text is converted into numbers that the model can process.

## Learning Goals

By the end of Week 1, you'll understand:
- How tokenization converts text to numbers
- The difference between character-level and subword tokenization
- How to prepare batched training data
- The concept of sequence length and context windows

## Tasks Breakdown

### Day 1-2: Setup & Understanding Tokenization

**Theory to Review:**
- What is tokenization and why we need it
- Character-level vs word-level vs subword (BPE, WordPiece)
- Vocabulary size trade-offs

**Practical Tasks:**
1. ✅ Project structure created
2. Install dependencies: `pip install -r requirements.txt`
3. Study `utils/tokenizer.py` - understand the CharTokenizer class
4. Test the tokenizer with sample text

**Exercise:**
```python
# Run this to test the tokenizer
python utils/tokenizer.py
```

### Day 3-4: Get Training Data

**Tasks:**
1. Download Shakespeare dataset (see `data/README.md`)
2. Read and explore the data
3. Build vocabulary from the dataset

**Exercise:**
Create a script `prepare_data.py`:
```python
from utils.tokenizer import CharTokenizer

# Load data
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset has {len(text)} characters")

# Build tokenizer
tokenizer = CharTokenizer()
tokenizer.build_vocab(text)

# Save tokenizer
tokenizer.save('data/tokenizer.json')

# Encode full dataset
tokens = tokenizer.encode(text)
print(f"Encoded to {len(tokens)} tokens")

# Sample
sample = text[:100]
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print(f"\nOriginal: {sample}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Match: {sample == decoded}")
```

### Day 5-6: Data Loader

**Theory to Review:**
- Why we use batches in training
- What is a sequence length/context window
- Input-output pairs for next-token prediction

**Tasks:**
1. Study `utils/data_loader.py`
2. Understand how TextDataset creates training examples
3. Test the data loader with your encoded data

**Exercise:**
```python
# Test data loading
from utils.tokenizer import CharTokenizer
from utils.data_loader import get_data_loader

# Load tokenizer and data
tokenizer = CharTokenizer()
tokenizer.load('data/tokenizer.json')

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenizer.encode(text)

# Create data loader
seq_len = 64  # Context length
batch_size = 32

loader = get_data_loader(tokens, seq_len, batch_size)

print(f"Number of training examples: {len(loader.dataset)}")
print(f"Number of batches: {len(loader)}")

# Examine one batch
x_batch, y_batch = next(iter(loader))
print(f"\nBatch shapes: X={x_batch.shape}, Y={y_batch.shape}")

# Decode first example to see what it looks like
first_x = x_batch[0].tolist()
first_y = y_batch[0].tolist()

print(f"\nFirst training example:")
print(f"Input : {tokenizer.decode(first_x)}")
print(f"Target: {tokenizer.decode(first_y)}")
```

### Day 7: Review & Prepare for Week 2

**Tasks:**
1. Review what you've learned
2. Make sure you understand:
   - How text → tokens → tensors
   - Why targets are shifted by 1 position
   - The role of sequence length
3. Read about attention mechanisms (preview for next week)

**Recommended Reading:**
- "The Illustrated Transformer" by Jay Alammar
- Review Section 3.2 of "Attention Is All You Need" paper
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (for context on sequence models)

## Week 1 Checklist

- [ ] Dependencies installed
- [ ] Tokenizer tested and understood
- [ ] Shakespeare dataset downloaded
- [ ] Vocabulary built from dataset
- [ ] Data loader tested
- [ ] Understand input-output pairs for training
- [ ] Ready to implement attention mechanism

## Questions to Answer Before Week 2

1. Why do we shift targets by 1 position?
2. What happens if we increase/decrease vocab size?
3. What is the trade-off of longer sequence length?
4. How does batch size affect training?

## Next Week Preview

Week 2 we'll implement the heart of the transformer: the attention mechanism! This is where the "magic" happens.

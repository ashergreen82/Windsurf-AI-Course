# Mini-GPT: Transformer Text Generator from Scratch

A educational implementation of a GPT-style transformer for text generation, built to understand how Large Language Models work.

## Project Overview

This project implements a character-level or word-level transformer model based on the "Attention Is All You Need" paper. The goal is to build a text generator from scratch using PyTorch.

## Learning Objectives

- Understand the Transformer architecture deeply
- Implement self-attention mechanisms
- Learn tokenization strategies
- Master training loops and optimization
- Explore text generation techniques

## Project Structure

```
mini-gpt/
├── data/                   # Training data and datasets
├── models/                 # Model architecture files
│   ├── transformer.py     # Main transformer model
│   ├── attention.py       # Attention mechanisms
│   └── layers.py          # Custom layers
├── utils/                  # Utility functions
│   ├── tokenizer.py       # Tokenization logic
│   ├── data_loader.py     # Data loading and batching
│   └── config.py          # Model configurations
├── train.py               # Training script
├── generate.py            # Text generation script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 4-Week Roadmap

### Week 1: Data & Tokenization (Nov 12-19)
- [ ] Set up project structure
- [ ] Implement character-level tokenizer
- [ ] Download and prepare training dataset (Shakespeare text)
- [ ] Create data loader with batching

### Week 2: Core Architecture (Nov 19-26)
- [ ] Implement scaled dot-product attention
- [ ] Build multi-head attention
- [ ] Create position-wise feed-forward network
- [ ] Implement positional encoding
- [ ] Assemble transformer blocks

### Week 3: Training (Nov 26-Dec 3)
- [ ] Implement training loop
- [ ] Add learning rate scheduling
- [ ] Implement gradient clipping
- [ ] Train first model on small dataset
- [ ] Add checkpointing and logging

### Week 4: Generation & Refinement (Dec 3-10)
- [ ] Implement greedy decoding
- [ ] Add temperature-based sampling
- [ ] Implement top-k and top-p sampling
- [ ] Create simple CLI interface
- [ ] Experiment and tune hyperparameters

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --epochs 50 --batch_size 64

# Generate text
python generate.py --prompt "To be or not to be" --length 200
```

## Model Architecture

- **Embedding Dimension**: 256
- **Number of Heads**: 8
- **Number of Layers**: 6
- **Feed-Forward Dimension**: 1024
- **Vocabulary Size**: ~50-10000 (depends on tokenizer)
- **Context Length**: 256 tokens

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Next Steps After Completion

- Fine-tune for specific tasks (chatbot, Q&A)
- Implement BERT-style bidirectional attention
- Scale up to larger datasets
- Explore computer vision (ViT - Vision Transformer)

## Notes

This is a learning project focused on understanding fundamentals. For production use, consider using pre-trained models from Hugging Face Transformers library.

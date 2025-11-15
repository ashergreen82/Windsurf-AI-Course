# Data Directory

This directory contains training data for the Mini-GPT model.

## Getting Started

### Option 1: Shakespeare Dataset (Recommended for learning)

Download the tiny Shakespeare dataset:
```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -OutFile "input.txt"

# Or manually download from:
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

This is a 1MB text file containing Shakespeare's works - perfect for learning!

### Option 2: Your Own Text

Place any `.txt` file here named `input.txt`. The model will learn from it.

## Dataset Information

- **Recommended size**: 1MB - 10MB for quick training
- **Format**: Plain text (.txt)
- **Encoding**: UTF-8
- **Content**: Any text works (books, articles, code, etc.)

## Current Files

Place `input.txt` in this directory to get started.

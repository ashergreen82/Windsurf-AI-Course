from utils.tokenizer import CharTokenizer
from pathlib import Path
import json

DATA_PATH = Path("data/input.txt")
TOKENIZER_PATH = Path("data/tokenizer.json")
TOKENS_PATH = Path("data/tokens.json")

def main():
    # Load raw text
    text = DATA_PATH.read_text(encoding="utf-8")
    print(f"Dataset has {len(text)} characters")

    # Build tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    # Encode full dataset
    tokens = tokenizer.encode(text)
    print(f"Encoded to {len(tokens)} tokens")

    # Save tokens (as JSON list for simplicity)
    TOKENS_PATH.write_text(json.dumps(tokens), encoding="utf-8")
    print(f"Saved tokens to {TOKENS_PATH}")

if __name__ == "__main__":
    main()
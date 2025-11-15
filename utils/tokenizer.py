"""
Simple character-level tokenizer for the Mini-GPT project.
We'll start with character-level to keep things simple, then can upgrade to BPE later.
"""

class CharTokenizer:
    """
    Character-level tokenizer that converts text to sequences of integers.
    Each unique character gets a unique integer ID.
    """
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, text):
        """
        Build vocabulary from text.
        
        Args:
            text (str): Training text to build vocabulary from
        """
        # Get unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        self.char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
        
        print(f"Vocabulary built with {self.vocab_size} characters")
        print(f"Characters: {''.join(chars)}")
    
    def encode(self, text):
        """
        Convert text to list of integers.
        
        Args:
            text (str): Text to encode
            
        Returns:
            list: List of integer token IDs
        """
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, tokens):
        """
        Convert list of integers back to text.
        
        Args:
            tokens (list): List of integer token IDs
            
        Returns:
            str: Decoded text
        """
        return ''.join([self.idx_to_char[idx] for idx in tokens if idx in self.idx_to_char])
    
    def save(self, filepath):
        """Save tokenizer vocabulary to file"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load(self, filepath):
        """Load tokenizer vocabulary from file"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
            self.vocab_size = data['vocab_size']


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = CharTokenizer()
    sample_text = "Hello, World! This is a test."
    tokenizer.build_vocab(sample_text)
    
    encoded = tokenizer.encode(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

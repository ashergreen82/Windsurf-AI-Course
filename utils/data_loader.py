"""
Data loading utilities for the Mini-GPT project.
Handles batching and sequence preparation.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Dataset for text generation.
    Creates input-output pairs for next-token prediction.
    """
    
    def __init__(self, tokens, seq_len):
        """
        Args:
            tokens (list): List of token IDs
            seq_len (int): Sequence length for training
        """
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        # Number of possible sequences
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        """
        Get a training example.
        Input: tokens[idx:idx+seq_len]
        Target: tokens[idx+1:idx+seq_len+1] (shifted by 1)
        """
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_data_loader(tokens, seq_len, batch_size, shuffle=True):
    """
    Create a DataLoader for training.
    
    Args:
        tokens (list): List of token IDs
        seq_len (int): Sequence length
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = TextDataset(tokens, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    return loader


if __name__ == "__main__":
    # Test the data loader
    sample_tokens = list(range(100))  # Mock tokens: [0, 1, 2, ..., 99]
    seq_len = 10
    batch_size = 4
    
    loader = get_data_loader(sample_tokens, seq_len, batch_size)
    
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    # Get one batch
    x, y = next(iter(loader))
    print(f"\nBatch shape - X: {x.shape}, Y: {y.shape}")
    print(f"First sequence X: {x[0].tolist()}")
    print(f"First sequence Y: {y[0].tolist()}")

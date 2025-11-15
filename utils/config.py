"""
Configuration file for the Mini-GPT transformer model.
Contains all hyperparameters and settings.
"""

class Config:
    """Model and training configuration"""
    
    # Model Architecture
    vocab_size = 100  # Will be set based on tokenizer
    d_model = 256  # Embedding dimension
    n_heads = 8  # Number of attention heads
    n_layers = 6  # Number of transformer blocks
    d_ff = 1024  # Feed-forward dimension
    max_seq_len = 256  # Maximum sequence length
    dropout = 0.1  # Dropout rate
    
    # Training
    batch_size = 64
    learning_rate = 3e-4
    epochs = 50
    warmup_steps = 4000
    gradient_clip = 1.0
    
    # Generation
    temperature = 0.8
    top_k = 40
    top_p = 0.9
    
    # Paths
    data_path = "data/input.txt"
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    # Device
    device = "cuda"  # Will be set to "cuda" if available, else "cpu"
    
    # Logging
    log_interval = 100  # Log every N batches
    save_interval = 1000  # Save checkpoint every N batches
    
    def __init__(self, **kwargs):
        """Allow overriding config values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def __str__(self):
        """Print configuration"""
        config_str = "Model Configuration:\n"
        config_str += "=" * 50 + "\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"{key:20s}: {value}\n"
        config_str += "=" * 50
        return config_str

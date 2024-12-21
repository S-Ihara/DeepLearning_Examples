from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class DefaultConfig:
    data_dir: str = "./dataset"
    log_dir: str = "./logs/logs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    patch_size: int = 4
    dim: int = 128
    num_heads: int = 8
    num_blocks: int = 4
    activation: nn.Module = nn.GELU()
    dropout: float = 0.1
    quiet_attention: bool = False


    batch_size: int = 2048
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 0.01

    num_workers: int = 4
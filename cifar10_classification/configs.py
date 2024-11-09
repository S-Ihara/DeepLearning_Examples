from dataclasses import dataclass

import torch

@dataclass
class DefaultConfig:
    data_dir: str = "./dataset"
    log_dir: str = "./logs/logs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size: int = 2048
    epochs: int = 40
    lr: float = 3e-4

    num_workers: int = 4
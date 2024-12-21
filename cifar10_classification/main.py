import argparse
from pathlib import Path
from datetime import datetime
from pprint import pformat

from tqdm import tqdm
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10

import configs 
from NN_Modules.modules import SimpleCNN
from core.train import train_loop
from core.eval import test_loop

try:
    from loguru import logger
except ImportError:
    import logging as logger

def main(config):
    logger.info("start CIFAR10 classification training")
    # dataset 
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.3),
        #transforms.Resize((32, 32)),
        transforms.ToDtype(torch.float32,scale=True),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    logger.info(f"dataset loading...")
    train_dataset = CIFAR10(root=config.data_dir, train=True, download=True, transform=compose)
    test_dataset = CIFAR10(root=config.data_dir, train=False, download=True, transform=compose)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # model
    model = SimpleCNN(num_classes=10)
    #model = resnet18(num_classes=10)
    model = model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    # 訓練
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    logger.info("start training")
    for epoch in range(config.epochs):
        train_info = train_loop(model=model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=config.device)
        train_losses.append(train_info["train_loss"])
        train_accuracies.append(train_info["train_acc"])
        logger.info(f"epoch: {epoch}, train_loss: {train_info['train_loss']}, train_acc: {train_info['train_acc']}")

        valid_info = test_loop(model=model, valid_loader=test_loader, loss_fn=loss_fn, device=config.device)
        valid_losses.append(valid_info["test_loss"])
        valid_accuracies.append(valid_info["test_acc"])
        logger.info(f"epoch: {epoch}, valid_loss: {valid_info['test_loss']}, valid_acc: {valid_info['test_acc']}")
        logger.info("---"*10)

    logger.info("finish training")

    # save model
    torch.save(model.state_dict(), f"{config.log_dir}/model.pth")
    logger.info(f"model saved at {config.log_dir}/model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="DefaultConfig")
    args = parser.parse_args()
    
    # log setting 
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = getattr(configs, args.config)(log_dir=f"./logs/{datetime_str}_{args.config}")
    Path(config.log_dir).mkdir(exist_ok=True, parents=True)
    logger.add(f"{config.log_dir}/info.log", rotation="10 MB")

    logger.info(f"config: {pformat(config.__dict__)}")

    # main(config)

    # # line profiler
    import line_profiler
    profile = line_profiler.LineProfiler()
    profile.add_function(main)
    profile.add_function(train_loop)
    profile.runcall(main, config)
    profile.print_stats(output_unit=1e-3)
    profile.dump_stats(f"{config.log_dir}/line_profiler.log")


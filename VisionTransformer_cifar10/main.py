import argparse
from pathlib import Path
from datetime import datetime
from pprint import pformat

from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import configs 
from nn_modules.vision_transformer import VisionTransformer
from core.train import train_loop
from core.eval import test_loop
from utilities.plot_graph import plot_loss, plot_accuracy

try:
    from loguru import logger
except ImportError:
    import logging as logger

def main(config):
    # clearml
    if getattr(config, "use_clearml", False):
        logger.info("use clearml")
        from clearml import Task
        now = datetime.now().strftime('%Y%m%d-%H%M%S')
        project_name = "vit_cifar10"
        task_name = "test"+"_"+now
        tag = "cifar10"
        comment = """
            Vision Transformer for CIFAR10 classification
        """
        task = Task.init(project_name=project_name, task_name=task_name)
        task.add_tags(tag)
        task.set_comment(comment)
        clearml_logger = task.get_logger()
        logger.info(f"clearml task: {task.id}")

        # log config
        task.connect(config, name="config")

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
    model = VisionTransformer(image_size=32, patch_size=config.patch_size, dim=config.dim, hidden_dim=config.dim*4, 
                              num_heads=config.num_heads, activation=config.activation, num_blocks=config.num_blocks,
                              dropout=config.dropout, quiet_attention=config.quiet_attention ,num_classes=10)
    model = model.to(config.device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
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

        if getattr(config, "use_clearml", False):
            clearml_logger.report_scalar(title="loss",series="train_loss",value=train_info["train_loss"],iteration=epoch)
            clearml_logger.report_scalar(title="acc",series="train_acc",value=train_info["train_acc"],iteration=epoch)
            clearml_logger.report_scalar(title="loss",series="valid_loss",value=valid_info["test_loss"],iteration=epoch)
            clearml_logger.report_scalar(title="acc",series="valid_acc",value=valid_info["test_acc"],iteration=epoch)
            clearml_logger.report_scalar(title="lr",series="lr",value=optimizer.param_groups[0]["lr"],iteration=epoch)

    logger.info("finish training")

    # plot graph
    train_losses = np.stack(train_losses)
    valid_losses = np.stack(valid_losses)
    train_accuracies = np.stack(train_accuracies)
    valid_accuracies = np.stack(valid_accuracies)
    plot_loss(config.log_dir, train_losses, valid_losses)
    plot_accuracy(config.log_dir, train_accuracies, valid_accuracies)

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
    if getattr(config, "line_profile", False):
        import line_profiler
        profile = line_profiler.LineProfiler()
        profile.add_function(main)
        profile.add_function(train_loop)
        profile.runcall(main, config)
        profile.print_stats(output_unit=1e-3)
        profile.dump_stats(f"{config.log_dir}/line_profiler.log")
    else:
        main(config)


import os
import shutil

from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
def plot_loss(save_dir: Path|str, train_losses: np.ndarray, valid_losses: np.ndarray|None=None, **kwargs):
    """損失を折れ線グラフで表示する
    args:
        save_dir(Path or str): 保存先ディレクトリ
        train_losses(np.ndarray[float]): shape=(n_epochs,)
        valid_losses(Optional[np.ndarray[float]]): shape=(n_epochs,)
        kwargs: その他の引数
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # check
    assert train_losses.ndim==1, f"train_losses.ndim={train_losses.ndim}"
    if valid_losses is not None:
        assert valid_losses.ndim==1, f"valid_losses.ndim={valid_losses.ndim}"

    sns.set_theme(style="darkgrid")

    fig,ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title("loss curve")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot(train_losses, label="train loss")
    if valid_losses is not None:
        ax.plot(valid_losses, label="valid loss")
    ax.legend()
    
    fig.savefig(save_dir/"loss.png")

    plt.rcParams.update(plt.rcParamsDefault) # seabornのスタイルをリセット


def plot_accuracy(save_dir: Path|str, train_accuracies: np.ndarray, valid_accuracies: np.ndarray|None=None, **kwargs):
    """精度を折れ線グラフで表示する
    args:
        save_dir(Path or str): 保存先ディレクトリ
        train_accuracies(np.ndarray[float]): shape=(n_epochs,)
        valid_accuracies(Optional[np.ndarray[float]]): shape=(n_epochs,)
        kwargs:
            fixed_ylim(bool): 0-1(or 0-100)でy軸の範囲を固定する
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # check
    assert train_accuracies.ndim==1, f"train_accuracies.ndim={train_accuracies.ndim}"
    if valid_accuracies is not None:
        assert valid_accuracies.ndim==1, f"valid_accuracies.ndim={valid_accuracies.ndim}"

    sns.set_theme(style="darkgrid")

    fig,ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title("accuracy curve")
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy")
    if kwargs.get("fixed_ylim", False):
        if train_accuracies.max() > 1:
            ax.set_ylim(0, 100.1)
        else:
            ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot(train_accuracies, label="train accuracy")
    if valid_accuracies is not None:
        ax.plot(valid_accuracies, label="valid accuracy")
    ax.legend()
    
    fig.savefig(save_dir/"accuracy.png")

    plt.rcParams.update(plt.rcParamsDefault) # seabornのスタイルをリセット
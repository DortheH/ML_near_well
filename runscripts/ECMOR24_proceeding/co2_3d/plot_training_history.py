"""
Plot the training history of the neural-network model.

The script reads the saved training-history CSV file and plots the training loss
and validation loss as functions of epoch. If validation loss is available, the
epoch with the lowest validation loss is marked in the figure. The loss axis can
be shown on a logarithmic scale to make changes during later training epochs more
visible. The resulting plot is saved as a figure for use in the thesis or
publication material.
"""

from __future__ import annotations

import pathlib
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_history(
    csv_path: str | pathlib.Path,
    savepath: str | pathlib.Path,
    log_y: bool = True,
) -> None:
    csv_path = pathlib.Path(csv_path)
    savepath = pathlib.Path(savepath)

    history = pd.read_csv(csv_path)

    if "epoch" in history.columns:
        epochs = history["epoch"] + 1
    else:
        epochs = range(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(epochs, history["loss"], label="Training loss", linewidth=1.5)

    if "val_loss" in history.columns:
        ax.plot(epochs, history["val_loss"], label="Validation loss", linewidth=1.5)

        best_idx = history["val_loss"].idxmin()
        best_epoch = int(epochs.iloc[best_idx])
        best_val = history["val_loss"].iloc[best_idx]

        ax.axvline(best_epoch, linestyle="--", linewidth=1)

        ax.annotate(
            f"Best val_loss\nEpoch {best_epoch}",
            xy=(best_epoch, best_val),
            xytext=(-45, 22),          # nærmere punktet enn før
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 0.8,
            },
        )

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("Training and validation loss during NN training", fontsize=11)

    ax.tick_params(axis="both", labelsize=8)

    ax.legend(
        loc="upper center",
        fontsize=8,
        frameon=True,
    )

    ax.grid(True, linewidth=0.6, alpha=0.7)

    if log_y:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    nn_dir = pathlib.Path("nn")
    plot_training_history(
        nn_dir / "training_history.csv",
        nn_dir / "training_validation_loss.png",
    )
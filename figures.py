"""Figure helpers shared by train.py.

Styling conventions: PNG output, no figure titles, legends outside the data
zone (above the axes), grid drawn behind the data, grey for the reference
series and green for the one that matters, key numbers in bold on a white box.
"""

import numpy as np

GREEN = "#2ca02c"
GREY = "#9e9e9e"
DARK = "#424242"


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color="#e0e0e0", linewidth=0.8)


def save_chip_grid(images, titles, colors, path, n_cols=5):
    """Grid of image chips, one short label per chip, no figure title."""
    import matplotlib.pyplot as plt

    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(2 * n_cols, 1.7 * n_rows))
    for i in range(n):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(images[i])
        ax.axis("off")
        ax.set_title(titles[i], fontsize=9, color=colors[i])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("Saved", path)
    return fig


def save_training_curves(history, path):
    """Accuracy, loss, and AUC per epoch: train in grey, validation in green.

    A dashed line marks the epoch with the best validation AUC (the one early
    stopping restores); its value sits in a white box in the top margin,
    outside the data zone.
    """
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmax(history["val_auc"]))
    panels = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (key, label) in zip(axes, panels):
        _style_axes(ax)
        ax.plot(epochs, history[key], color=GREY, linewidth=1.6)
        ax.plot(epochs, history["val_" + key], color=GREEN, linewidth=1.6)
        ax.axvline(epochs[best], color=GREY, linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
    fig.legend(["training", "validation"], loc="upper center", ncol=2,
               frameon=False)
    fig.text(0.99, 0.96, "best val AUC %.3f, epoch %d"
             % (history["val_auc"][best], epochs[best]),
             ha="right", va="center", fontsize=9, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="#bdbdbd",
                       boxstyle="round,pad=0.35"))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    print("Saved", path)
    return fig


def save_confusion_matrix(cm, class_names, path):
    """2x2 confusion matrix, blue shading by share of the true class."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    row_share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(row_share, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if row_share[i, j] > 0.6 else DARK
            ax.text(j, i, "%d\n%.1f%%" % (cm[i, j], 100 * row_share[i, j]),
                    ha="center", va="center", color=color, fontweight="bold")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("Saved", path)
    return fig

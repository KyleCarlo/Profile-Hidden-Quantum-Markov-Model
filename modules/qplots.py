"""Plots for HQMM mini-batch COSM training.

History data comes from modules.qtrain.train_hqmm:
    iter, train_time, loglik,
    val_da_mean, val_da_std, test_da_mean, test_da_std

The `train_time` column is cumulative training-batch wall-clock time,
not CPU process time; it excludes the initial/end-of-epoch evaluations.
"""

import matplotlib.pyplot as plt
import numpy as np


def _references(ax, references):
    """Add optional horizontal reference lines to an existing axis."""
    styles = ("--", ":", "-.")
    for i, (label, value) in enumerate((references or {}).items()):
        ax.axhline(
            y=float(value),
            linestyle=styles[i % len(styles)],
            color="0.25",
            linewidth=1.2,
            label=label,
            zorder=1,
        )


def plot_da_learning_curve(histories, references=None, x="train_time", out="q_da_vs_time.png"):
    """Plot test description accuracy during HQMM COSM training.

    The shaded band is mean +/- one *population* standard deviation
    across test sequences (as recorded by qtrain.py).

    Parameters
    ----------
    histories : dict[str, pandas.DataFrame]
        Mapping from step-size labels to the histories from train_hqmm().
    references : dict[str, float] | None
        Optional horizontal benchmark DA values.
    x : {'train_time', 'iter'}
        The horizontal axis: cumulative *training* seconds or epochs.
    out : str or pathlib.Path
        Destination PNG filename.
    """
    if x not in ("train_time", "iter"):
        raise ValueError("x must be 'train_time' or 'iter'")
    if not histories:
        raise ValueError("histories must contain at least one training run")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for label, h in histories.items():
        required = {x, "train_da_mean", "train_da_std"}
        missing = required.difference(h.columns)
        if missing:
            raise ValueError(f"History for {label!r} is missing columns: {sorted(missing)}")

        times = h[x].to_numpy(dtype=float)
        means = h["train_da_mean"].to_numpy(dtype=float)
        stds = h["train_da_std"].to_numpy(dtype=float)

        (line,) = ax.plot(
            times, means,
            marker="o", markersize=3,
            linewidth=1.8,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            times,
            means - stds,
            means + stds,
            color=line.get_color(),
            alpha=0.18,
            zorder=2,
        )

    _references(ax, references)

    ax.set_xlabel(
        "Training time (s; evaluation excluded)"
        if x == "train_time"
        else "Training epoch"
    )
    ax.set_ylabel("Test description accuracy (DA)")
    ax.set_title("HQMM Learning Curves (COSM)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_da_bars(results, references=None, out="q_da_bars.png"):
    """Plot final test DA by HQMM COSM step size, with +/- 1 std error bars.

    `results` maps labels (including optionally 'untrained') to
    (test_da_mean, test_da_std) tuples. All values may be Python scalars
    or scalar PyTorch tensors.
    """
    if not results:
        raise ValueError("results must contain at least one model")

    labels = list(results)
    means = np.array([float(results[label][0]) for label in labels], dtype=float)
    stds = np.array([float(results[label][1]) for label in labels], dtype=float)

    trained_labels = [label for label in labels if label != "untrained"]
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        "0.65" if label == "untrained"
        else default_colors[trained_labels.index(label) % len(default_colors)]
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(max(6, 1.45 * len(labels) + 1.5), 4.5))
    positions = np.arange(len(labels))
    ax.bar(
        positions, means,
        yerr=stds,
        capsize=5,
        width=0.65,
        color=colors,
        error_kw={"elinewidth": 1.2, "ecolor": "0.2"},
        zorder=2,
    )
    _references(ax, references)

    bounds = [float(v) for v in (references or {}).values()]
    lo = min([float(np.min(means - stds))] + bounds)
    hi = max([float(np.max(means + stds))] + bounds)
    span = max(hi - lo, 0.02)
    ax.set_ylim(lo - 0.10 * span, hi + 0.25 * span)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test description accuracy (DA)")
    ax.set_title("Final HQMM Test DA by COSM step size")
    if references:
        ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

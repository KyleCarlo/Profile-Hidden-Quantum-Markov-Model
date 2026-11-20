
"""Plots for HQMM mini-batch COSM training.

History data comes from modules.qtrain.train_hqmm:
    iter, train_time, loglik,
    train_da_mean,
    val_da_mean, val_da_std,
    test_da_mean, test_da_std

The `train_time` column is cumulative training-batch wall-clock time,
not CPU process time; it excludes the initial/end-of-epoch evaluations.
"""

import matplotlib.pyplot as plt
import numpy as np


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color="#b0b0b0", alpha=0.25, linewidth=1)

    # Full rectangular border around the plot
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#404040")
        spine.set_linewidth(1.2)

    ax.tick_params(axis="both", labelsize=11, width=1, length=5)


def _references(ax, references):
    """Add optional horizontal reference lines to an existing axis."""
    styles = ("--", ":", "-.")
    for i, (label, value) in enumerate((references or {}).items()):
        ax.axhline(
            y=float(value),
            linestyle=styles[i % len(styles)],
            color="0.25",
            linewidth=1.5,
            label=label,
            zorder=1,
        )


def plot_da_learning_curve(
    histories,
    references=None,
    x="train_time",
    out="q_da_vs_time.png",
    title="PHQMM Learning Curves (COSM)",
):
    """DA vs training time (or epoch) during HQMM COSM training.

    Dashed lines = training DA.
    Solid lines = validation DA.
    Shaded bands = validation mean +/- 1 population std.

    Parameters
    ----------
    histories : dict[str, pandas.DataFrame]
        Mapping from step-size labels to training histories.
    references : dict[str, float] | None
        Optional horizontal benchmark DA values.
    x : {'train_time', 'iter'}
        Cumulative training seconds or epochs.
    out : str or pathlib.Path
        Destination PNG filename.
    title : str
        Title of the plot.
    """
    if x not in ("train_time", "iter"):
        raise ValueError("x must be 'train_time' or 'iter'")
    if not histories:
        raise ValueError("histories must contain at least one training run")

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, h in histories.items():
        required = {
            x,
            "train_da_mean",
            "val_da_mean",
            "val_da_std",
        }
        missing = required.difference(h.columns)
        if missing:
            raise ValueError(
                f"History for {label!r} is missing columns: {sorted(missing)}"
            )

        times = h[x].to_numpy(dtype=float)
        train_means = h["train_da_mean"].to_numpy(dtype=float)
        val_means = h["val_da_mean"].to_numpy(dtype=float)
        val_stds = h["val_da_std"].to_numpy(dtype=float)

        # Training DA (dashed, no standard deviation)
        (line,) = ax.plot(
            times,
            train_means,
            ls="--",
            lw=2.6,
            label=f"{label} (Training)",
            zorder=3,
        )

        color = line.get_color()

        # Validation DA (solid)
        ax.plot(
            times,
            val_means,
            ls="-",
            lw=2.6,
            color=color,
            label=f"{label} (Validation)",
            zorder=4,
        )

        # Validation standard deviation only
        ax.fill_between(
            times,
            val_means - val_stds,
            val_means + val_stds,
            color=color,
            alpha=0.13,
            zorder=2,
        )

    _references(ax, references)
    _style_axes(ax)

    ax.set_xlabel(
        "Training time (s; evaluation excluded)"
        if x == "train_time"
        else "Epoch",
        fontsize=13,
        labelpad=8,
    )
    ax.set_ylabel(
        "Description Accuracy (DA)",
        fontsize=13,
        labelpad=8,
    )
    ax.set_title(title, fontsize=17, pad=12)

    ax.legend(
        loc="lower right",
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_da_bars(
    results,
    references=None,
    out="q_da_bars.png",
    title="Final HQMM Test DA by COSM step size",
):
    """Plot final test DA by HQMM COSM step size.

    Bars = mean test DA.
    Error bars = +/- 1 std across test sequences.

    Results map labels (including optionally 'untrained')
    to (test_da_mean, test_da_std) tuples.
    """
    if not results:
        raise ValueError("results must contain at least one model")

    labels = list(results)

    means = np.array(
        [float(results[label][0]) for label in labels],
        dtype=float,
    )
    stds = np.array(
        [float(results[label][1]) for label in labels],
        dtype=float,
    )

    trained_labels = [
        label for label in labels if label != "untrained"
    ]
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    colors = [
        "0.65" if label == "untrained"
        else default_colors[
            trained_labels.index(label) % len(default_colors)
        ]
        for label in labels
    ]

    fig, ax = plt.subplots(
        figsize=(max(7, 1.45 * len(labels) + 1.5), 5.5)
    )

    positions = np.arange(len(labels))

    ax.bar(
        positions,
        means,
        yerr=stds,
        capsize=5,
        width=0.65,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
        error_kw={
            "elinewidth": 1.5,
            "ecolor": "0.2",
            "capthick": 1.5,
        },
        zorder=2,
    )

    _references(ax, references)
    _style_axes(ax)

    ax.grid(axis="x", visible=False)

    bounds = [
        float(v) for v in (references or {}).values()
    ]
    lo = min([float(np.min(means - stds))] + bounds)
    hi = max([float(np.max(means + stds))] + bounds)

    span = max(hi - lo, 0.02)

    ax.set_ylim(
        lo - 0.10 * span,
        hi + 0.25 * span,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels,
        rotation=15,
        ha="right",
    )

    ax.set_ylabel(
        "Test DA",
        fontsize=13,
        labelpad=8,
    )
    ax.set_title(title, fontsize=17, pad=12)

    if references:
        ax.legend(
            loc="upper left",
            fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.95,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

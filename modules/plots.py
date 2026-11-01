
import numpy as np
import matplotlib.pyplot as plt


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color="#b0b0b0", alpha=0.25, linewidth=1)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", labelsize=11, width=1, length=5)


def _references(ax, references):
    for i, (label, value) in enumerate((references or {}).items()):
        ax.axhline(
            value,
            ls=["--", ":", "-."][i % 3],
            color="0.25",
            lw=1.5,
            label=label,
            zorder=1,
        )


def plot_da_learning_curve(
    histories,
    references=None,
    x="cpu_time",
    out="da_vs_time.png",
    title="Description Accuracy during Training",
):
    """DA vs CPU training time (or epoch).
    Dashed lines = training DA, solid lines = validation DA.
    Shaded bands = validation mean +/- 1 std across sequences."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, h in histories.items():
        (line,) = ax.plot(
            h[x],
            h["train_da_mean"],
            ls="--",
            lw=2.6,
            label=f"{label} (Training)",
            zorder=3,
        )

        color = line.get_color()

        ax.plot(
            h[x],
            h["val_da_mean"],
            ls="-",
            lw=2.6,
            color=color,
            label=f"{label} (Validation)",
            zorder=4,
        )

        ax.fill_between(
            h[x],
            h["val_da_mean"] - h["val_da_std"],
            h["val_da_mean"] + h["val_da_std"],
            color=color,
            alpha=0.13,
            zorder=2,
        )

    _references(ax, references)
    _style_axes(ax)

    ax.set_xlabel(
        "CPU time (s)" if x == "cpu_time" else "Epoch",
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
    out="da_bars.png",
    title="Test DA by prior strength",
):
    """Fig. 1 style: test DA of each model; error bars = 1 std of DA across test sequences.
    Like the paper, the y-axis is zoomed to the range of the results."""

    labels = list(results)
    means = np.array([results[k][0] for k in labels])
    stds = np.array([results[k][1] for k in labels])

    fig, ax = plt.subplots(
        figsize=(max(7, 1.3 * len(labels) + 1.5), 5.5)
    )

    trained = [k for k in labels if k != "untrained"]
    colors = [
        "0.65" if k == "untrained" else f"C{trained.index(k)}"
        for k in labels
    ]

    ax.bar(
        range(len(labels)),
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        width=0.6,
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

    lo = min(
        (means - stds).min(),
        *(references or {}).values()
    ) if references else (means - stds).min()

    hi = max(
        (means + stds).max(),
        *(references or {}).values()
    ) if references else (means + stds).max()

    ax.set_ylim(
        lo - 0.1 * (hi - lo),
        hi + 0.25 * (hi - lo),
    )

    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Test DA", fontsize=13, labelpad=8)
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

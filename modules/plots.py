import numpy as np
import matplotlib.pyplot as plt

def _references(ax, references):
    for i, (label, value) in enumerate((references or {}).items()):
        ax.axhline(value, ls=["--", ":", "-."][i % 3], color="0.25", lw=1.2, label=label, zorder=1)


def plot_da_learning_curve(histories, references=None, x="cpu_time", out="da_vs_time.png"):
    """Fig. 2 style: test DA vs CPU training time (or x="iter"). The line is the mean over
    test sequences and the band is +/- 1 std across test sequences."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, h in histories.items():
        (line,) = ax.plot(h[x], h["train_da_mean"], marker="o", ms=3, lw=1.8, label=label)
        ax.fill_between(h[x], h["train_da_mean"] - h["train_da_std"],
                        h["train_da_mean"] + h["train_da_std"], color=line.get_color(), alpha=0.18)
    _references(ax, references)
    ax.set_xlabel("CPU time (s)" if x == "cpu_time" else "EM iteration")
    ax.set_ylabel("Train DA")
    ax.set_title("Train DA during Baum-Welch training")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_da_bars(results, references=None, out="da_bars.png"):
    """Fig. 1 style: test DA of each model; error bars = 1 std of DA across test sequences.
    Like the paper, the y-axis is zoomed to the range of the results."""
    labels = list(results)
    means = np.array([results[k][0] for k in labels])
    stds = np.array([results[k][1] for k in labels])
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(labels) + 1.5), 4.2))
    trained = [k for k in labels if k != "untrained"]           # same colours as plot_da_vs_time
    colors = ["0.65" if k == "untrained" else f"C{trained.index(k)}" for k in labels]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=5, color=colors, width=0.6,
           error_kw={"elinewidth": 1.2, "ecolor": "0.2"}, zorder=2)
    _references(ax, references)
    lo = min((means - stds).min(), *(references or {}).values()) if references else (means - stds).min()
    hi = max((means + stds).max(), *(references or {}).values()) if references else (means + stds).max()
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.25 * (hi - lo))   # headroom for the legend
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Test DA")
    ax.set_title("Test DA by prior strength")
    if references:
        ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

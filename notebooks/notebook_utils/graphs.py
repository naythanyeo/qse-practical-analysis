"""Graphing functions"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import HTML, display
from io import BytesIO
import base64


ANSATZ_ORDER = ["UCCSD", "UCCSDSinglet", "UCCGSD", "1UpCCGSDSinglet"]


def plot_spin_eigenvalue_spread(singlet_data, triplet_data, title=None, ax_labels=True):
    available_ansatz = list(dict.fromkeys([*singlet_data.keys(), *triplet_data.keys()]))
    ansatz_names = [ansatz for ansatz in ANSATZ_ORDER if ansatz in available_ansatz]
    ansatz_names.extend(ansatz for ansatz in available_ansatz if ansatz not in ansatz_names)
    colors = dict(zip(ansatz_names, sns.color_palette("tab10", n_colors=len(ansatz_names))))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, data, spin_label in zip(axes, [singlet_data, triplet_data], ["Singlet", "Triplet"]):
        for ansatz in ansatz_names:
            points = data.get(ansatz, [])
            if not points:
                continue

            x_values, eigenvalues = zip(*points)
            ax.scatter(
                x_values,
                eigenvalues,
                s=16,
                alpha=0.7,
                color=colors[ansatz],
                label=ansatz,
            )

        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e1)
        ax.set_title(spin_label)
        ax.grid(True, which="both", alpha=0.25)

        if ax_labels:
            ax.set_xlabel("Eigenvalue Index")

    if ax_labels:
        axes[0].set_ylabel("Overlap eigenvalue")

    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(dict(zip(labels, handles)))

    if handles_by_label:
        fig.legend(
            handles_by_label.values(),
            handles_by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(handles_by_label), 4),
            frameon=False,
        )

    if title is not None:
        fig.suptitle(title, y=0.96, fontsize=15)

    fig.tight_layout(rect=(0, 0.16, 1, 0.94))

    return fig, axes


def show_scrollable_figs(figs, max_height=750, max_width="100%", dpi=150):
    max_height = f"{max_height}px" if isinstance(max_height, (int, float)) else max_height
    max_width = f"{max_width}px" if isinstance(max_width, (int, float)) else max_width

    images = []
    for fig in figs:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)

        encoded = base64.b64encode(buf.getvalue()).decode()
        images.append(f"""
        <img
            src="data:image/png;base64,{encoded}"
            style="display: block; max-width: 100%; height: auto; margin-bottom: 16px;"
        >
        """)

    display(HTML(f"""
    <div style="max-height: {max_height}; max-width: {max_width}; overflow-y: auto; overflow-x: hidden; border: 1px solid #ccc;">
        {''.join(images)}
    </div>
    """))

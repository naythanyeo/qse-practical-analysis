"""
Graphing functions used throughout the notebooks
Graphing functions will return a figure object
Within notebook controls the plotting or saving of that figure
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from IPython.display import HTML, display
from io import BytesIO
from scipy.stats import linregress
import base64

from notebook_utils.chem import CHEMICAL_ACCURACY


def show_scrollable_figs(figs, max_height=750, max_width="100%", dpi=150):
    """
    Helper function to display all plots as scrollabe elements in the
    jupyter notebooks
    """
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

# ========================================================
# Notebook 1: Representability and Spectral Accuracy
# ========================================================

def plot_tiered_error_ridgelines(error_data):
    """
    Main Fig 1
    Plot UCCSD low-state error distributions as a ridgeline plot 
    Show the error distributions acros active spaces 
    """
    states = ["s0", "s1", "t1", "t2"]
    active_spaces = list(error_data)
    positions = np.arange(len(active_spaces))[::-1]
    log_limits = (-12, 2.5)
    bin_edges = np.linspace(*log_limits, 90)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    smoothing_kernel = np.array([1, 4, 6, 4, 1]) / 16
    chemical_accuracy_mha = 1000 * CHEMICAL_ACCURACY
    tier_bounds = np.log10([chemical_accuracy_mha, 10, 50])
    tier_colors = ["#59A14F", "#EDC948", "#F28E2B", "#E15759"]
    tier_labels = [
        f"$\\leq$ chemical accuracy ({chemical_accuracy_mha:.2f} mHa)",
        "1.6-10 mHa",
        "10-50 mHa",
        "> 50 mHa",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    for axis, state in zip(axes.flat, states):
        for position, active_space in zip(positions, active_spaces):
            errors = np.asarray(error_data[active_space][state], dtype=float)
            if not len(errors):
                continue

            log_errors = np.log10(np.maximum(errors, 1e-12))
            density, _ = np.histogram(log_errors, bins=bin_edges, density=True)
            density = np.convolve(density, smoothing_kernel, mode="same")
            density = 0.72 * density / density.max()
            tier_masks = [
                bin_centres <= tier_bounds[0],
                (bin_centres > tier_bounds[0]) & (bin_centres < tier_bounds[1]),
                (bin_centres >= tier_bounds[1]) & (bin_centres <= tier_bounds[2]),
                bin_centres > tier_bounds[2],
            ]

            for mask, color in zip(tier_masks, tier_colors):
                axis.fill_between(
                    bin_centres, position, position + density, where=mask,
                    interpolate=True, color=color, alpha=0.82,
                )

            axis.plot(bin_centres, position + density, color="white", linewidth=0.6)
            axis.scatter(np.median(log_errors), position + 0.04, color="black", s=13, zorder=3)

        axis.axvline(tier_bounds[0], color="black", linestyle=":", linewidth=1.1)
        axis.set_xlim(*log_limits)
        axis.set_xticks([-12, -9, -6, -3, 0, 2],
                        [rf"$10^{{{tick}}}$" for tick in [-12, -9, -6, -3, 0, 2]])
        axis.set_yticks(positions, active_spaces)
        axis.set_ylim(-0.5, len(active_spaces) - 0.02)
        axis.set_title(state.upper())
        axis.grid(axis="x", alpha=0.2)

    for axis in axes[:, 0]:
        axis.set_ylabel("Active space")

    handles = [Patch(facecolor=color, label=label) for color, label in zip(tier_colors, tier_labels)]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
                  ncol=5, frameon=False)
    figure.supxlabel("Raw errors (mHa)", y=0.09)
    figure.suptitle(
        "Low-lying state error distributions across active space (UCCSD)",
        y=0.98,
        fontsize=18
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.91, wspace=0.11, hspace=0.20)
    return figure


def plot_projection_error_scatter(projection_error_data):
    """Plot low-lying UCCSD errors against exact-state representability deficit."""
    states = ["s0", "s1", "t1", "t2"]
    active_spaces = projection_error_data["active_space"].drop_duplicates().tolist()
    colors = plt.colormaps["viridis"](np.linspace(0.08, 0.92, len(active_spaces)))
    chemical_accuracy_mha = 1000 * CHEMICAL_ACCURACY
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for state, axis in zip(states, axes.flat):
        state_data = projection_error_data[projection_error_data["state"] == state]
        for active_space, color in zip(active_spaces, colors):
            active_data = state_data[state_data["active_space"] == active_space]
            axis.scatter(
                np.maximum(active_data["deficit_percent"], 1e-7), active_data["error_mHa"],
                color=color, s=30, edgecolor="white", linewidth=0.35, alpha=0.82,
            )

        axis.axvline(1, color="#B24C4C", linestyle=":", linewidth=1.0)
        axis.axhline(chemical_accuracy_mha, color="black", linestyle=":", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(state)
        axis.grid(which="both", alpha=0.14)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="white", label=active_space)
        for active_space, color in zip(active_spaces, colors)
    ]
    handles.extend([
        Line2D([0], [0], color="#B24C4C", linestyle=":", label="Projection P = 0.99"),
        Line2D([0], [0], color="black", linestyle=":", label="Chemical accuracy"),
    ])
    figure.supxlabel("Exact-state representability deficit, 100(1 - P) (%)", y=0.19)
    figure.supylabel("Absolute index-matched energy error (mHa)")
    figure.suptitle("Large energy errors need not imply low exact-state projection", y=0.98)
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                  ncol=5, frameon=False)
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.86,
                           wspace=0.08, hspace=0.14)
    return figure


def plot_s1_active_space_error(error_data, molecule):
    """Plot one molecule's indexed UCCSD S1 error across active spaces."""
    case = error_data[error_data["molecule"] == molecule]
    active_spaces = case["active_space"].tolist()
    errors = case["error_mHa"].to_numpy(dtype=float)
    positions = np.arange(len(active_spaces))
    chemical_accuracy_mha = 1000 * CHEMICAL_ACCURACY

    figure, axis = plt.subplots(figsize=(5.2, 3.7))
    axis.plot(positions, np.maximum(errors, 1e-4), color="#D1495B", marker="o", linewidth=2.2)
    axis.axhline(chemical_accuracy_mha, color="black", linestyle=":", linewidth=1.0)
    axis.set_yscale("log")
    axis.set_ylim(1e-4, 220)
    axis.set_xticks(positions, active_spaces, rotation=45, ha="right")
    axis.set_xlabel("Active space")
    axis.set_ylabel("Indexed S1 error (mHa)")
    axis.set_title(molecule)
    axis.grid(axis="y", which="both", alpha=0.18)
    figure.tight_layout()
    return figure



def plot_q1_best_root_transition_map(overlap_data):
    """
    Plot q1's closest exact-singlet root across molecules and active spaces.
    """
    molecules = overlap_data["molecule"].drop_duplicates().tolist()
    active_spaces = overlap_data["active_space"].drop_duplicates().tolist()
    root_colors = {"s1": "#4E79A7", "s2": "#F28E2B", "other": "#59A14F"}
    cell_colors = np.empty((len(molecules), len(active_spaces), 3))

    figure, axis = plt.subplots(figsize=(13, 11.5))
    for row_index, molecule in enumerate(molecules):
        molecule_data = overlap_data[overlap_data["molecule"] == molecule].set_index("active_space")
        for column_index, active_space in enumerate(active_spaces):
            record = molecule_data.loc[active_space]
            color_key = record["best_exact_root"] if record["best_exact_root"] in root_colors else "other"
            strength = 0.15 + 0.85 * record["overlap_percent"] / 100
            cell_colors[row_index, column_index] = 1 - strength * (1 - np.array(to_rgb(root_colors[color_key])))

    axis.imshow(cell_colors, aspect="auto")
    axis.set_xticks(np.arange(len(active_spaces)), active_spaces, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(molecules)), molecules)
    axis.set_xlabel("Active space")
    axis.set_ylabel("Molecule")
    axis.set_title("Direct q1 best-match root across active space")

    for row_index, molecule in enumerate(molecules):
        molecule_data = overlap_data[overlap_data["molecule"] == molecule].set_index("active_space")
        for column_index, active_space in enumerate(active_spaces):
            record = molecule_data.loc[active_space]
            text_color = "white" if record["overlap_percent"] > 55 else "black"
            axis.text(
                column_index,
                row_index,
                f"{record['best_exact_root']}\n{record['overlap_percent']:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.tight_layout()
    return figure


def plot_bad_spin_roots(bad_spin_roots, active_space, threshold=0.01):
    """
    Fig SI 1
    Input: Bad spin roots count in a dictionary, active space 
    Plot bad-spin root counts for one active space, separated by expansion
    Active space and threshold for title label
    """
    ansatzes = list(bad_spin_roots)
    colors = sns.color_palette("colorblind", len(ansatzes))
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for axis, expansion in zip(axes, ("singlet", "triplet")):
        root_labels = list(bad_spin_roots[ansatzes[0]][expansion])
        positions = np.arange(len(root_labels))
        bottom = np.zeros(len(root_labels))

        for ansatz, color in zip(ansatzes, colors):
            counts = [len(bad_spin_roots[ansatz][expansion][root]) for root in root_labels]
            axis.bar(positions, counts, bottom=bottom, width=0.85, color=color, label=ansatz)
            bottom += counts

        tick_step = max(1, len(root_labels) // 12)
        tick_positions = positions[::tick_step]
        axis.set_xticks(tick_positions, [root_labels[index] for index in tick_positions])
        axis.set_title(expansion.title())
        axis.set_xlabel("QSE root")
        axis.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Total bad roots")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                  ncol=3, frameon=False)
    figure.suptitle(
        rf"{active_space}: spin-contaminated roots ($|\Delta S^2| > {threshold}$)", y=1.04
    )
    sns.despine()
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    return figure

# ========================================================
# Notebook 2: Dimensions of QSE Subspace
# ========================================================
def plot_spin_eigenvalue_spread(eigenvalue_data, active_space):
    """
    Plot UCCSD singlet and triplet overlap eigenvalue spreads.
    Input: Dicitionary of {singlet: (points), triplet: (points)}
    Split 2 panel plot, log y axis scale 
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for axis, expansion in zip(axes, ("singlet", "triplet")):
        indices, eigenvalues = zip(*eigenvalue_data[expansion])
        axis.scatter(indices, eigenvalues, s=16, color="black", alpha=0.55)
        axis.set_yscale("log")
        axis.set_ylim(1e-14, 1e2)
        axis.set_title(expansion.title())
        axis.set_xlabel("Eigenvalue index")
        axis.grid(True, which="both", alpha=0.2)

    axes[0].set_ylabel("Overlap eigenvalue")
    figure.suptitle(f"UCCSD {active_space} Overlap Eigenvalue Spread")
    figure.tight_layout()
    return figure


def plot_triplet_eigenvalue_spread(eigenvalue_points, active_space):
    """
    Plot the 1UpCCGSDSinglet triplet overlap eigenvalue spread
    Same as before but only singlet panel graph 
    """
    indices, eigenvalues = zip(*eigenvalue_points)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(indices, eigenvalues, s=16, color="black", alpha=0.55)
    axis.set_yscale("log")
    axis.set_ylim(1e-14, 1e2)
    axis.set_title(f"1UpCCGSDSinglet {active_space} Triplet Overlap Eigenvalue Spread")
    axis.set_xlabel("Eigenvalue index")
    axis.set_ylabel("Overlap eigenvalue")
    axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    return figure


def plot_eigenvector_composition(composition_data, active_space):
    """
    Plot mean QSE operator composition for singlet and triplet eigenvectors.
    Input: {singlet: {0: {Number occupied: [percentage], ...}}
            triplet: {0: ...}}
    Plots as a two panel eigenvector stacked bar chart
    """
    categories = ["Number occupied", "OV", "Rest"]
    # Legends at the bottom
    category_labels = {
        "Number occupied": "Number Operator (Occupied)",
        "OV": "Occupied to Virtual Operators",
        "Rest": "Other Operators",
    }
    colors = sns.color_palette("colorblind", len(categories))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for axis, expansion in zip(axes, ("singlet", "triplet")):
        ranks = list(composition_data[expansion])
        positions = np.arange(len(ranks))
        bottom = np.zeros(len(ranks))

        for category, color in zip(categories, colors):
            percentages = [composition_data[expansion][rank][category] for rank in ranks]
            axis.bar(positions, percentages, bottom=bottom, width=0.9,
                     color=color, label=category_labels[category])
            bottom += percentages

        tick_step = max(1, len(ranks) // 12)
        tick_positions = positions[::tick_step]
        axis.set_xticks(tick_positions, [ranks[index] for index in tick_positions])
        axis.set_ylim(0, 100)
        axis.set_title(expansion.title())
        axis.set_xlabel("Eigenvector rank")
        axis.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Mean composition (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                  ncol=3, frameon=False)
    figure.suptitle(f"UCCSD {active_space} Overlap Eigenvector Composition", y=0.98)
    sns.despine()
    figure.tight_layout(rect=(0, 0.12, 1, 0.92))
    return figure


def plot_dimension_error_distribution(dimension_errors):
    """
    Compare maximum and stable QSE-dimension errors by low-lying state.
    Input: Dictionary containing all the errors
    Plots max vs stable states box plot, coloured 
    """
    states = ("s0", "s1", "t1", "t2")
    methods = {
        "max_states": ("Max states", "tab:blue"),
        "stable_states": ("Stable states", "tab:orange"),
    }
    positions = np.arange(len(states))
    figure, axis = plt.subplots(figsize=(9, 6))

    for offset, (method, (label, color)) in zip((-0.2, 0.2), methods.items()):
        boxes = axis.boxplot(
            [dimension_errors[method][state] for state in states],
            positions=positions + offset,
            widths=0.35,
            patch_artist=True,
            boxprops={"facecolor": color, "edgecolor": color},
            medianprops={"color": "black"},
            whiskerprops={"color": color},
            capprops={"color": color},
            flierprops={"markeredgecolor": color},
        )
        boxes["boxes"][0].set_label(label)

    axis.set_xticks(positions, states)
    axis.set_ylabel("Absolute energy error (mHa)")
    axis.set_title("UCCSD 6e6o: Maximum and Stable QSE Dimensions")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    sns.despine()
    figure.tight_layout()
    return figure


# ========================================================
# Notebook 3: Shot Noise and Thresholding
# ========================================================
def plot_threshold_shot_errors(error_data, thresholds):
    """
    Plot mean QSE errors against overlap threshold for each shot count
    Input: Error data is a dictionry of form {shot: [mean errors]}
    Total 41 thresholds used from the threshold ladder
    """
    colors = sns.color_palette("viridis", len(error_data))
    figure, axis = plt.subplots(figsize=(9, 6))

    for (n_shots, errors), color in zip(error_data.items(), colors):
        label = f"{n_shots // 1000}k" if n_shots < 1_000_000 else "1M"
        axis.plot(thresholds, errors, marker="o", markersize=4, linewidth=1.5,
                  color=color, label=label)

    axis.axhline(1000 * CHEMICAL_ACCURACY, color="black", linestyle=":",
                 label="Chemical accuracy")
    axis.set_xscale("log")
    axis.set_ylim(0, 20)
    axis.set_xlabel("Overlap eigenvalue threshold")
    axis.set_ylabel("Mean absolute energy error (mHa)")
    axis.set_title("Mean Error Against Threshold and Shot Count")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                ncol=6, frameon=False, title="Shot count", fontsize=9,
                handlelength=1.5, columnspacing=1.2, labelspacing=0.4)
    sns.despine()
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    return figure


def format_latex_scientific(value):
    coefficient, exponent = f"{value:.1e}".split("e")
    return rf"${coefficient} \times 10^{{{int(exponent)}}}$"


def plot_best_threshold_heatmap(best_threshold_data):
    """
    Plot the mean optimal overlap threshold by active space and shot count
    Input: Dicionary of dictionaries
    Plots heatmap from that
    """
    threshold_df = pd.DataFrame.from_dict(best_threshold_data, orient="index")
    annotations = threshold_df.map(format_latex_scientific)
    shot_labels = [f"{n_shots // 1000}k" if n_shots < 1_000_000 else "1M"
                   for n_shots in threshold_df.columns]

    figure, axis = plt.subplots(figsize=(8.5, 6))
    sns.heatmap(
        threshold_df, cmap="Blues", norm=LogNorm(vmin=1e-4, vmax=1),
        annot=annotations, annot_kws={"fontsize": 8}, fmt="", linewidths=0.5, ax=axis,
        cbar_kws={"label": "Mean optimal overlap threshold"},
    )
    axis.set_xticklabels(shot_labels, rotation=45, ha="right")
    axis.set_xlabel("Shot count")
    axis.set_ylabel("Active space")
    axis.set_title("Mean Optimal Threshold by Active Space and Shot Count")
    figure.tight_layout()
    return figure


def plot_error_scaling_regression(scaling_points):
    """
    Plot mean shot errors against the orbital and shot-count scaling coordinate
    Input list of coordinates
    Plot and do linear regression
    """
    x, y = np.asarray(scaling_points, dtype=float).T
    regression = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 200)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(x, y, color="black", alpha=0.65)
    axis.plot(x_fit, regression.intercept + regression.slope * x_fit, color="black")
    axis.text(0.04, 0.95, rf"$R^2 = {regression.rvalue**2:.3f}$",
              transform=axis.transAxes, va="top")
    axis.set_xlabel(r"$N_{\mathrm{orb}}^2 / \sqrt{N_{\mathrm{shots}}}$")
    axis.set_ylabel("Mean minimum energy error (mHa)")
    axis.set_title("Shot-Noise Scaling of Minimum QSE Error")
    axis.grid(alpha=0.2)
    sns.despine()
    figure.tight_layout()
    return figure


def plot_shot_eigenvalue_spread(eigenvalue_data, active_space, n_shots):
    """
    Plot statevector and sampled overlap eigenvalue spreads.
    Input dictionary of {Singlet: {SV: [eigenvalsh], Shots: [eigenvalsh]...}}
    Plots in blue (SV) and red (shots) with negative axis shown
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for axis, expansion in zip(axes, ("singlet", "triplet")):
        shot_indices, shot_eigenvalues = zip(*eigenvalue_data[expansion]["shots"])
        sv_indices, sv_eigenvalues = zip(*eigenvalue_data[expansion]["statevector"])
        shot_eigenvalues = np.where(np.abs(shot_eigenvalues) < 1e-14, 0, shot_eigenvalues)
        sv_eigenvalues = np.where(np.abs(sv_eigenvalues) < 1e-14, 0, sv_eigenvalues)

        axis.scatter(sv_indices, sv_eigenvalues, s=16, color="blue", alpha=0.7,
                     label="Statevector", zorder=2)
        axis.scatter(shot_indices, shot_eigenvalues, s=12, color="red", alpha=0.3,
                     label="Shots", zorder=3)
        axis.axhline(0, color="black", linewidth=1.5, zorder=4)
        axis.set_yscale("symlog", linthresh=1e-14, linscale=2)
        axis.set_ylim(-1e-2, 1e2)
        axis.set_title(expansion.title())
        axis.set_xlabel("Eigenvalue index")
        axis.grid(True, which="both", alpha=0.2)

    axes[0].set_ylabel("Overlap eigenvalue")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(f"UCCSD {active_space} Overlap Eigenvalue Lifting ({n_shots:,} shots)")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    return figure


def plot_correlated_error_distribution(error_data):
    """Plot mean low-state errors after preserving or breaking H-S pairing."""
    conditions = ["Matched noisy H/S", "Shifted noisy H/S", "Noisy H / statevector S"]
    labels = ["Matched\nnoisy H/S", "Shifted\nnoisy H/S", "Noisy H /\nstatevector S"]
    distributions = [
        error_data.loc[error_data["condition"] == condition, "mean_error_mHa"].to_numpy()
        for condition in conditions
    ]

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    boxes = axis.boxplot(
        distributions,
        vert=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": "black", "markersize": 3},
    )
    for box in boxes["boxes"]:
        box.set(facecolor="white", edgecolor="black")

    axis.set_xscale("log")
    axis.set_yticks(range(1, len(labels) + 1), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Mean absolute low-state error vs statevector (mHa)")
    axis.set_title("Effect of Breaking H-S Correlated Errors: UCCSD 6e6o, 100k Shots, 1e-2 Threshold")
    axis.grid(axis="x", which="both", alpha=0.2)
    sns.despine()
    figure.tight_layout()
    return figure


# ========================================================
# Notebook 4: Scaling and Heatmaps
# ========================================================
def plot_h_s_element_regression(points):
    """
    Scatter plot of the Hij and Sij correlation
    Input: list of points (Hij and Sij)
    Scatter plots them and does linear regression
    """
    s_elements, h_elements = np.asarray(points, dtype=float).T
    retained = (s_elements > 1e-12) & (h_elements > 1e-12)
    s_elements = s_elements[retained]
    h_elements = h_elements[retained]

    regression = linregress(np.log10(s_elements), np.log10(h_elements))
    fit_s = np.logspace(np.log10(s_elements.min()), np.log10(s_elements.max()), 300)
    fit_h = 10 ** regression.intercept * fit_s ** regression.slope

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter(s_elements, h_elements, s=5, alpha=0.08, color="tab:blue",
                 linewidths=0, rasterized=True)
    axis.plot(fit_s, fit_h, color="tab:red", linewidth=2)
    axis.set(xscale="log", yscale="log", xlabel=r"$|S_{ij}|$", ylabel=r"$|H_{ij}|$",
             title="Corresponding Off-Diagonal QSE Matrix Elements")
    axis.text(
        0.04, 0.96,
        rf"$R^2 = {regression.rvalue**2:.4f}$",
        transform=axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    axis.grid(True, which="both", alpha=0.2)
    return figure


def plot_reordered_overlap_heatmaps(matrix_data, active_space, expansion):
    """
    Plot 28 reordered overlap matrices
    Input: Dictionary of sorted matrix data 
    Plots regular heatmap for each of them (7 by 4)
    """
    figure, axes = plt.subplots(4, 7, figsize=(20, 12), layout="constrained")
    log_matrices = []
    for _, matrix, _ in matrix_data:
        magnitude = np.abs(matrix)
        log_matrix = np.full(magnitude.shape, np.nan)
        retained = magnitude >= 1e-12
        log_matrix[retained] = np.log10(magnitude[retained])
        log_matrices.append(log_matrix)
    vmin = min(np.nanmin(matrix) for matrix in log_matrices)
    vmax = max(np.nanmax(matrix) for matrix in log_matrices)
    colormap = sns.color_palette("magma", as_cmap=True)
    colormap.set_bad("black")

    for axis, (molecule, _, _), log_matrix in zip(axes.flat, matrix_data, log_matrices):
        image = axis.imshow(log_matrix, cmap=colormap, vmin=vmin, vmax=vmax, aspect="equal")
        axis.set_title(molecule, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.colorbar(image, ax=axes, shrink=0.72, label=r"$\log_{10}|S_{ij}|$")
    figure.suptitle(
        f"UCCSD {active_space} {expansion.title()} Overlap Matrices",
    )
    return figure


def plot_shot_overlap_heatmaps(matrix_data, active_space):
    """
    Plots the same heatmaps but for shots data
    Input: Dictionary of matrix data, by molecule 
    3 molecules only from the data, each molecule has SV and shots S
    Plots the same heatmaps from before
    """
    molecules = list(matrix_data)
    columns = list(matrix_data[molecules[0]])
    figure, axes = plt.subplots(3, 5, figsize=(14, 9), layout="constrained")

    log_matrices = []
    for matrices in matrix_data.values():
        for matrix in matrices.values():
            magnitude = np.abs(matrix)
            log_matrix = np.full(magnitude.shape, np.nan)
            retained = magnitude >= 1e-12
            log_matrix[retained] = np.log10(magnitude[retained])
            log_matrices.append(log_matrix)

    vmin = min(np.nanmin(matrix) for matrix in log_matrices)
    vmax = max(np.nanmax(matrix) for matrix in log_matrices)
    colormap = sns.color_palette("magma", as_cmap=True)
    colormap.set_bad("black")

    for row, molecule in enumerate(molecules):
        for column, label in enumerate(columns):
            matrix = matrix_data[molecule][label]
            magnitude = np.abs(matrix)
            log_matrix = np.full(magnitude.shape, np.nan)
            retained = magnitude >= 1e-12
            log_matrix[retained] = np.log10(magnitude[retained])
            image = axes[row, column].imshow(
                log_matrix, cmap=colormap, vmin=vmin, vmax=vmax, aspect="equal"
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

            if row == 0:
                axes[row, column].set_title(label)
            if column == 0:
                axes[row, column].set_ylabel(molecule)

    figure.colorbar(image, ax=axes, shrink=0.82, label=r"$\log_{10}|S_{ij}|$")
    figure.suptitle(f"UCCSD {active_space} Singlet Overlap Matrices")
    return figure


def plot_commuting_group_scaling(commuting_group_data):
    """
    Plot mean commuting groups against the number of spatial orbitals
    INPUT: commuting_group_data, dictionary of singlet and triplet keys
    Plots data and gets the regression curve 
    """
    fig, axes = plt.subplots(1, 2, 
                             sharey=True,
                             sharex=True,
                             layout="constrained")
    for axis, expansion in zip(axes, commuting_group_data.keys()):
        group_data = commuting_group_data[expansion]

        num_orbitals = np.asarray(sorted(group_data), dtype=float)
        num_groups = np.asarray([group_data[num_orbital] for num_orbital in num_orbitals])
        regression = linregress(np.log10(num_orbitals), np.log10(num_groups))
        fit_orbitals = np.logspace(np.log10(num_orbitals.min()), np.log10(num_orbitals.max()), 200)
        fit_groups = 10 ** regression.intercept * fit_orbitals ** regression.slope

        axis.scatter(num_orbitals, num_groups, color="black", s=45, zorder=2)
        axis.plot(fit_orbitals, fit_groups, color="black", linewidth=1.5, zorder=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(num_orbitals, [str(int(value)) for value in num_orbitals])
        axis.set_title(f"{"Singlet" if expansion=="singlet" else "Triplet"}")
        axis.text(
            0.04, 0.96,
            rf"$N_{{\mathrm{{groups}}}} \propto n^{{{regression.slope:.2f}}}$"
            "\n"
            rf"$R^2 = {regression.rvalue**2:.4f}$",
            transform=axis.transAxes, va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )
        axis.grid(True, which="both", alpha=0.2)
    fig.suptitle("Commuting-Group Scaling")
    fig.supylabel("Mean number of commuting groups")
    fig.supxlabel("Number of Spatial Orbitals")
    return fig

# ========================================================
# Notebook 6: Molecular Orbitals
# ========================================================
def plot_low_state_double_weight_scaling(double_weight_data):
    """Plot exact S1, T1, and T2 double-excitation weights across active spaces."""
    states = ["s1", "t1", "t2"]
    labels = {"s1": "S1", "t1": "T1", "t2": "T2"}
    colors = {"s1": "#E15759", "t1": "#4E79A7", "t2": "#59A14F"}
    active_spaces = list(double_weight_data)
    positions = np.arange(len(active_spaces))
    figure, axis = plt.subplots(figsize=(10, 5.5))

    for state in states:
        values = [np.asarray(double_weight_data[active_space].get(state, []), dtype=float)
                  for active_space in active_spaces]
        medians = [np.median(value) if len(value) else np.nan for value in values]
        lower = [np.percentile(value, 25) if len(value) else np.nan for value in values]
        upper = [np.percentile(value, 75) if len(value) else np.nan for value in values]

        axis.fill_between(positions, lower, upper, color=colors[state], alpha=0.16)
        axis.plot(positions, medians, marker="o", linewidth=2.1,
                  color=colors[state], label=labels[state])

    axis.set_xticks(positions, active_spaces, rotation=40, ha="right")
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Active space")
    axis.set_ylabel("Double-excitation weight (%)")
    axis.set_title("CASCI double-excitation character across active space")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(title="Exact state", frameon=False)
    figure.tight_layout()
    return figure


def plot_s1_casci_composition(composition_data, molecule):
    """Plot one molecule's exact-CASCI S1 excitation composition by active space."""
    categories = ("singles", "paired_doubles", "mixed_doubles", "higher")
    colors = {
        "singles": "#4E79A7",
        "paired_doubles": "#E15759",
        "mixed_doubles": "#F28E2B",
        "higher": "#B07AA1",
    }
    labels = {
        "singles": "Singles",
        "paired_doubles": "Paired doubles",
        "mixed_doubles": "Mixed doubles",
        "higher": "Higher",
    }
    case = composition_data[composition_data["molecule"] == molecule]
    active_spaces = case["active_space"].tolist()
    positions = np.arange(len(active_spaces))

    figure, axis = plt.subplots(figsize=(5.2, 3.7))
    bottom = np.zeros(len(case))
    for category in categories:
        values = case[category].to_numpy(dtype=float)
        axis.bar(positions, values, bottom=bottom, color=colors[category], label=labels[category], width=0.72)
        bottom += values

    axis.set_ylim(0, 100)
    axis.set_xticks(positions, active_spaces, rotation=45, ha="right")
    axis.set_xlabel("Active space")
    axis.set_ylabel("Exact CASCI weight (%)")
    axis.set_title(molecule)
    axis.grid(axis="y", alpha=0.18)
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    return figure


# ========================================================
# Notebook 7: Selected Doubles
# ========================================================
def plot_ov_doubles_s1_error_scatter(pool_comparison):
    """Compare 6e6o UCCSD S1 errors before and after adding OV doubles."""
    chemical_accuracy_mha = 1000 * CHEMICAL_ACCURACY
    limits = np.array([0.04, 250])
    label_offsets = {
        "Pyrrole": (5, -4), "Furan": (6, 0), "Benzene": (3, 3),
        "Hexatriene": (-2, 4), "Octatetraene": (-5, -5),
    }

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(
        pool_comparison["singles_only_mHa"], pool_comparison["ov_doubles_mHa"],
        color="black", edgecolor="white", linewidth=0.75, s=58, alpha=0.92, zorder=3,
    )
    axis.plot(limits, limits, color="black", linestyle=":", linewidth=1.25, zorder=2)
    axis.axvline(chemical_accuracy_mha, color="black", linestyle="--", linewidth=0.9)
    axis.axhline(chemical_accuracy_mha, color="black", linestyle="--", linewidth=0.9)
    axis.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
    axis.set_xlabel("Singles-only S1 error (mHa)")
    axis.set_ylabel("OV-double S1 error (mHa)")
    axis.set_title("Effect of adding OV-Doubles Excitation Operators")
    axis.grid(which="both", alpha=0.17)

    for molecule, offset in label_offsets.items():
        row = pool_comparison.loc[pool_comparison["molecule"] == molecule].iloc[0]
        axis.annotate(
            molecule, (row["singles_only_mHa"], row["ov_doubles_mHa"]),
            xytext=offset, textcoords="offset points", fontsize=8,
            ha="right" if offset[0] < 0 else "left",
        )

    figure.tight_layout()
    return figure

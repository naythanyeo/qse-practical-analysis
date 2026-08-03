"""
Graphing functions used throughout the notebooks
Graphing functions will return a figure object
Within notebook controls the plotting or saving of that figure
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
def plot_error_distribution(error_data):
    """
    Fig S1: Plot error distribution of the SV calculations
    INPUT: error_data dictionary containing relevant info
    PLOTS: 4 panel boxplot showing the error distributions
    """
    states = ["s0", "s1", "t1", "t2"]
    ansatzes = list(error_data)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    for state, axis in zip(states, axes.flat):
        state_errors = [error_data[ansatz][state] for ansatz in ansatzes]
        axis.boxplot(
            state_errors, tick_labels=ansatzes,
            boxprops={"color": "black"}, medianprops={"color": "black"},
            whiskerprops={"color": "black"}, capprops={"color": "black"},
            flierprops={"markeredgecolor": "black"},
        )
        axis.set_title(state)
        axis.set_ylabel("Absolute energy error (mHa)")
        axis.tick_params(axis="x", rotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    figure.tight_layout()
    return figure


def plot_projection_error(projection_error_data):
    """
    Fig 2?
    Plot exact-state projection against QSE energy error
    Input: Dictionary of (projection, error) for each root type
    Plots: 4 Panel scatter plot 
    """
    states = ["s0", "s1", "t1", "t2"]
    projection_ticks = np.round(np.arange(0.90, 1.001, 0.01), 2)
    tick_labels = ["<0.9", *[f"{value:.2f}" for value in projection_ticks[1:-1]], "1.0"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for state, axis in zip(states, axes.flat):
        projections, errors = zip(*projection_error_data[state])
        axis.scatter(projections, errors, color="black", alpha=0.65)
        axis.axhline(1000 * CHEMICAL_ACCURACY, color="black", linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(0.895, 1.005)
        axis.set_xticks(projection_ticks, tick_labels, rotation=45)
        axis.set_title(state)
        axis.set_xlabel("Exact-state projection")
        axis.set_ylabel("Absolute energy error (mHa)")

    figure.suptitle("UCCSD 6e6o Error and Projection Spread")
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

def plot_active_space_error_iqr(error_data):
    """
    Fig 3??
    Plot median UCCSD errors and their interquartile ranges by active space.
    Input: Dictionary of dictionaries  with spread of errors
    Plots: IQR range only 
    """
    states = ["s0", "s1", "t1", "t2"]
    active_spaces = list(error_data)
    positions = np.arange(len(active_spaces))
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for state, axis in zip(states, axes.flat):
        state_errors = [error_data[active_space][state] for active_space in active_spaces]
        medians = np.asarray([np.median(values) for values in state_errors])
        quartiles = np.asarray([np.percentile(values, [25, 75]) for values in state_errors])
        iqr = np.vstack((medians - quartiles[:, 0], quartiles[:, 1] - medians))

        axis.errorbar(medians, positions, xerr=iqr, fmt="o", color="black", capsize=4)
        axis.axvline(1000 * CHEMICAL_ACCURACY, color="black", linestyle=":")
        axis.set_xscale("log")
        axis.set_yticks(positions, active_spaces)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(state)
        axis.set_xlabel("Absolute energy error (mHa)")
        axis.grid(axis="x", alpha=0.2)

    axes[0, 0].invert_yaxis()
    figure.suptitle("UCCSD Error Scaling Across Active Spaces")
    figure.tight_layout()
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


def plot_best_threshold_heatmap(best_threshold_data):
    """
    Plot the mean optimal overlap threshold by active space and shot count
    Input: Dicionary of dictionaries
    Plots heatmap from that
    """
    threshold_df = pd.DataFrame.from_dict(best_threshold_data, orient="index")
    annotations = threshold_df.map(lambda value: f"{value:.1e}")
    shot_labels = [f"{n_shots // 1000}k" if n_shots < 1_000_000 else "1M"
                   for n_shots in threshold_df.columns]

    figure, axis = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        threshold_df, cmap="Blues", norm=LogNorm(vmin=1e-4, vmax=1),
        annot=annotations, fmt="", linewidths=0.5, ax=axis,
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

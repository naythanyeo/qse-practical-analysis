"""
Helper functions for latex formatting
"""

import re
from pathlib import Path

import pandas as pd


REPLACEMENTS = {
    "±": lambda _: r"$\pm$",
    r"(?<![\w.])([-+]?\d+(?:\.\d+)?)e([+-]?\d+)(?!\w)": (
        lambda match: rf"${match[1]} \times 10^{{{int(match[2])}}}$"
    ),
}

GROUND_STATE_COLUMN_FORMAT = (
    "l "
    "S[table-format=3.0] "
    "S[table-format=5.0] "
    "S[table-format=2.3] @{\\,\\(\\pm\\)\\,} S[table-format=3.3] "
    "S[table-format=1.4] @{\\,\\(\\pm\\)\\,} S[table-format=1.4] "
    "S[table-format=1.3e2] @{\\,\\(\\pm\\)\\,} S[table-format=1.3e2]"
)


def pandas2tex(df, header, caption, output_path, label=None, font_size=None, column_spacing=None):
    """
    Convert pandas dataframe into latex table 
    Input df and Header / Captions / labels etc 
    Formats the output nicely, saves to output path
    """
    header = tuple(header)
    expected_headers = len(df.columns) + 1

    if len(header) != expected_headers:
        raise ValueError(f"Expected {expected_headers} headers, received {len(header)}.")

    pandas_header = [value.replace("{", "{{").replace("}", "}}") for value in header]
    latex_table = df.reset_index().to_latex(
        index=False,
        header=pandas_header,
        caption=caption,
        label=label,
        position="htbp",
        column_format=f"l{'r' * len(df.columns)}",
        escape=True,
    )
    # Replace special characters and scientific notation
    for pattern, replacement in REPLACEMENTS.items():
        latex_table = re.sub(pattern, replacement, latex_table)

    table_start = "\\begin{table}[htbp]\n\\centering"
    if font_size is not None:
        table_start += f"\n\\{font_size}"
    if column_spacing is not None:
        table_start += f"\n\\setlength{{\\tabcolsep}}{{{column_spacing}pt}}"

    latex_table = latex_table.replace(r"\begin{table}[htbp]", table_start, 1)

    # Save the file 
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table, encoding="utf-8")


def ground_state2tex(df, caption, output_path, label=None):
    """Write a ground-state comparison dataframe with aligned numeric columns."""
    centered = lambda value: rf"\multicolumn{{1}}{{c}}{{{value}}}"
    format_value = lambda value, spec: "" if pd.isna(value) else format(value, spec)

    error_mean = [format_value(values[0], ".3f") for values in df["mean_error"]]
    error_max = [format_value(values[1], ".3f") for values in df["mean_error"]]
    projection_mean = [format_value(values[0], ".4f") for values in df["mean_projection"]]
    projection_max = [format_value(values[1], ".4f") for values in df["mean_projection"]]
    spin_mean = [format_value(values[0], ".3e") for values in df["spin_contamination"]]
    spin_max = [format_value(values[1], ".3e") for values in df["spin_contamination"]]

    table = pd.DataFrame({
        "ansatz": df.index,
        "parameters": df["num_parameters"].astype(int),
        "depth": df["circuit_depth"].astype(int),
        "error_mean": error_mean,
        "error_max": error_max,
        "projection_mean": projection_mean,
        "projection_max": projection_max,
        "spin_mean": spin_mean,
        "spin_max": spin_max,
    })
    table.columns = pd.MultiIndex.from_tuples([
        ("Ansatz", ""),
        (centered("Parameters"), ""),
        (centered("Circuit depth"), ""),
        ("Error (mHa)", centered("Mean")),
        ("Error (mHa)", centered("Max. diff.")),
        ("Projection", centered("Mean")),
        ("Projection", centered("Max. diff.")),
        ("Spin contamination", centered("Mean")),
        ("Spin contamination", centered("Max. diff.")),
    ])

    latex_table = table.to_latex(
        index=False,
        caption=caption,
        label=label,
        position="htbp",
        column_format=GROUND_STATE_COLUMN_FORMAT,
        multicolumn=True,
        multicolumn_format="c",
        escape=False,
    )
    table_start = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}"
    )
    latex_table = latex_table.replace(r"\begin{table}[htbp]", table_start, 1)
    latex_table = latex_table.replace(
        r"\begin{tabular}",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
        1,
    )
    latex_table = latex_table.replace(r"\end{tabular}", "\\end{tabular}\n}", 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table, encoding="utf-8")

"""Various helper plotting functions that make data analysis easier."""
import itertools
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

# Support Python 3.7 by importing Literal from typing_extensions
try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from . import flow

WellMappingStyle = Union[Literal["category"], Literal["linear"], Literal["log"], Dict[str, Any]]


def plot_mapping(
    mapping: Dict[str, Any],
    *,
    plate_size: Optional[Tuple[int, int]] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
    style: Optional[WellMappingStyle] = None,
):
    """
    Plot a single well mapping as projected onto a 96 well plate.

    Parameters
    ----------
    mapping
        The mapping from well names to condition values to plot.
    plate_size
        The width and height of the plate, in number of wells.
        Defaults to a 96- or 384-well plate, depending on the size of the mapping.
    fig
        A matplotlib Figure to use to plot on. If not specified, a new figure is created.

    Returns
    -------
    A `matplotlib.figure.Figure` object encoding the plot.
    """
    if fig is None:
        fig = plt.figure()
    ax = fig.subplots()  # type: ignore

    if plate_size is None:
        # Check if we have any wells beyond H or beyond 12
        if any([k[0] > "H" or int(k[1:]) > 12 for k in mapping.keys()]):
            plate_size = (24, 16)
        else:
            plate_size = (12, 8)

    # Only use the two-digit mappings so they sort properly
    sorted_mapping_values = [
        t[1]
        for t in sorted(
            [tup for tup in mapping.items() if len(tup[0]) == 3], key=lambda tup: tup[0]
        )
    ]
    mapping_vals = sorted(set(sorted_mapping_values), key=lambda x: sorted_mapping_values.index(x))
    if style is None:
        # Autodetect mapping.
        # If it is all numbers, then we find the median. If all numbers are non-negative
        # and the median lies outside [.15, .85] percentile ranges, guess log scale.
        if not all([isinstance(x, (int, float)) for x in mapping_vals]):
            style = "category"
        else:
            min_val = min(mapping_vals)
            max_val = max(mapping_vals)

            if min_val < 0 or max_val == min_val:
                style = "linear"
            else:
                median = np.median(mapping_vals)
                median_percentile = (median - min_val) / (max_val - min_val)
                if median_percentile < 0.15 or median_percentile > 0.85:
                    style = "log"
                else:
                    style = "linear"
    if style == "category":
        base_cmap = plt.get_cmap("rainbow")  # type: ignore
        mapping_colors = {
            val: base_cmap(idx / len(mapping_vals)) for idx, val in enumerate(mapping_vals)
        }
    elif style == "linear":
        mapping_vals = sorted(mapping_vals)
        min_val = min(mapping_vals)
        max_val = max(mapping_vals)
        base_cmap = plt.get_cmap("viridis")  # type: ignore
        mapping_colors = {
            x: base_cmap((x - min_val) / (max_val - min_val) * 0.85) for x in mapping_vals
        }
    elif style == "log":
        mapping_vals = sorted(mapping_vals)
        min_nonzero_val = min([x for x in mapping_vals if x > 0])
        max_val = max(mapping_vals)
        base_cmap = plt.get_cmap("viridis")  # type: ignore
        mapping_colors = {}
        for x in mapping_vals:
            if x == 0:
                mapping_colors[x] = "k"
            else:
                mapping_colors[x] = base_cmap(
                    (np.log(x) - np.log(min_nonzero_val))
                    / (np.log(max_val) - np.log(min_nonzero_val))
                    * 0.85
                )
    else:
        mapping_colors = style

    row_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for row, col in itertools.product(range(plate_size[1]), range(plate_size[0])):
        well = row_str[row] + str(col + 1)

        ax.add_patch(
            plt.Circle(
                (col, -row),
                0.45,
                ec="k",
                fc=mapping_colors[mapping[well]] if well in mapping else None,
            )
        )
    # Create the legend
    legend_entries = [
        plt.Circle((0, 0), 0.45, ec="k", fc=color) for color in mapping_colors.values()
    ]
    ax.legend(legend_entries, mapping_colors.keys(), loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-0.5, -0.5 + plate_size[0]])
    ax.set_ylim([0.5 - plate_size[1], 0.5])
    ax.set_xticks(range(plate_size[0]), labels=range(1, plate_size[0] + 1))
    ax.set_yticks(
        [-i for i in range(plate_size[1])], labels=[row_str[i] for i in range(plate_size[1])]
    )
    fig.tight_layout()


def plot_well_metadata(
    filename: Union[pathlib.Path, str],
    *,
    output_dir: Optional[pathlib.Path] = None,
    plate_size: Optional[Tuple[int, int]] = None,
    columns: Optional[List[str]] = None,
    style: Optional[Dict[str, WellMappingStyle]] = None,
):
    """
    Plot the specified metadata columns listed in a YAML file.

    Parameters
    ----------
    filename: `str` or `pathlib.Path`
        The path to the YAML file containing the mapping
    output_dir: optional `pathlib.Path`
        If given, outputs plate maps as PNGs, PDFs, and SVGs into this folder.
        If not given, plots are `plt.show`'d interactively.
    plate_size: optional `Tuple[int,int]`
        The width and height of the plate, in number of wells.
        Defaults to a 96- or 384-well plate.
    columns: optional `List[str]`
        The list of columns to plot. If not specified, all metadata columns are plotted.
    """
    mapping = flow.load_well_metadata(filename)
    if columns is None:
        columns = list(mapping.keys())
    for column in columns:
        col_style = None if (style is None or column not in style) else style[column]
        plot_mapping(mapping[column], plate_size=plate_size, style=col_style)
        plt.title(column)
        if output_dir is None:
            plt.show()
        else:
            plt.savefig(output_dir / f"{column}.png", bbox_inches="tight")
            plt.savefig(output_dir / f"{column}.svg", bbox_inches="tight")
            plt.savefig(output_dir / f"{column}.pdf", bbox_inches="tight")
        plt.close()

"""Various helper plotting functions that make data analysis easier."""

import itertools
import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.colors
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        if any(k[0] > "H" or int(k[1:]) > 12 for k in mapping.keys()):
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
        if not all(isinstance(x, (int, float)) for x in mapping_vals):
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
            x: base_cmap((x - min_val) / (max_val - min_val) * 0.85 if min_val != max_val else 0.0)
            for x in mapping_vals
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
                fc=mapping_colors[mapping[well]] if well in mapping else "#00000000",
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


def generate_xticklabels(
    df_labels: pd.DataFrame,
    ax_col,
    label_cols: List,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    align_ticklabels: Optional[
        Union[Literal["left"], Literal["center"], Literal["right"]]
    ] = "center",
    align_annotation: Optional[
        Union[Literal["left"], Literal["center"], Literal["right"]]
    ] = "right",
    linespacing: Optional[float] = 1.2,
):
    """
    Create table-like x-axis tick labels based on provided metadata.

    Parameters
    ----------
    df_labels: Pandas DataFrame
        DataFrame of metadata related to original xticklabels. Columns are metadata values,
        including the x-axis value.
    ax_col:
        Column of 'df_labels' that contains the original xticklabels. For seaborn plots, this
        should be equivalent to the column passed to the x variable.
    label_cols:
        List of columns of 'df_labels' to use to replace the xticklabels.
    ax: Optional Matplotlib axes
        Axes to edit, uses current axes if none specified.
    align_ticklabels: 'left', 'center', or 'right'
        Text alignment for multi-line xticklabels.
    align_annotation: 'left', 'center', or 'right'
        Text alignment for multi-line annotations comprising the columns of 'df_labels' used to
        replace the xticklabels. These appear to the bottom left of the plot, with the bounding
        box right-aligned with the right of the yticklabels and aligned vertically center with
        the vertical center of the xticklabels.
    linespacing: float
        Spacing between rows of the new xticklabels, as a multiple of the font size. Keeps
        matplotlib default (1.2) if not specified.

    Returns
    -------
    None; modifies the axes in place.

    """
    # Draw plotting canvas to generate original xticklabels
    if ax is None:
        ax = plt.gca()
    ax.figure.canvas.draw()

    # Create dictionary from DataFrame, where keys are original xticklabels
    #  and values are dictionaries of (metadata_key, metadata_value) pairs
    dict_labels_by_xticklabel = df_labels.set_index(ax_col).to_dict(orient="index")

    # Loop over xticklabels and set new values
    ax_labels = []
    for item in ax.get_xticklabels():
        if item.get_text() in dict_labels_by_xticklabel:
            dict_labels = dict_labels_by_xticklabel[item.get_text()]

            # For each specified metadata key (label_cols), get the metadata value
            #  and concatenate all values into separate lines of a single string
            new_xticklabel = "\n".join([str(dict_labels[i]) for i in label_cols])
            ax_labels.append(new_xticklabel)

        # If the original label is not in the provided dictionary, leave as is
        else:
            ax_labels.append(item.get_text())

    ax.set_xticks(
        ax.get_xticks(), ax_labels, multialignment=align_ticklabels, linespacing=linespacing
    )

    # Add annotation with metadata keys, only if the plot has yticklabels
    if ax.get_yticklabels():
        # Get Artists for first axes labels
        xlabel_bbox = ax.get_xticklabels()[0]
        ylabel_bbox = ax.get_yticklabels()[0]

        font_size = plt.rcParams["xtick.labelsize"]

        # Annotate plot with metadata keys
        #   x value: align the right of the annotation bbox (ha='right')
        #       with the right (x=1) of the ylabel bbox (xcoord=ylabel_bbox)
        #   y value: align the vertical center of the annotation bbox (va='center')
        #       with the vertical center (y=0.5) of the xlabel bbox (ycoord=xlabel_bbox)
        ax.annotate(
            text="\n".join(label_cols),
            xy=(1, 0.5),
            xycoords=(ylabel_bbox, xlabel_bbox),
            ha="right",
            va="center",
            multialignment=align_annotation,
            fontsize=font_size,
            linespacing=linespacing,
        )


def adjust_subplot_margins_inches(
    subfig: matplotlib.figure.SubFigure,
    *,
    left=0.0,
    bottom=0.0,
    top=0.0,
    right=0.0,
):
    """
    Adjust subplot margins to specified margins in inches.

    This adjusts the extent of all subplot axes are
    placed the specified number of inches from the edges of the subfigure.

    Parameters
    ----------
    fig: matplotlib SubFigure
        Subfigure to be adjusted
    left: float
        Left margin, in inches
    bottom: float
        Bottom margin, in inches
    top: float
        Top margin, in inches
    right: float
        Right margin, in inches

    Returns
    -------
    None; modifies the subfigure in place.

    """
    to_inches = subfig.transSubfigure + subfig.dpi_scale_trans.inverted()
    to_subfig = to_inches.inverted()
    # Calculate subfigure offsets
    lb_corner = to_subfig.transform(to_inches.transform([0.0, 0.0]) + [left, bottom])
    rt_corner = to_subfig.transform(to_inches.transform([1.0, 1.0]) - [right, top])
    subfig.subplots_adjust(
        left=lb_corner[0], bottom=lb_corner[1], right=rt_corner[0], top=rt_corner[1]
    )


def debug_axes(fig: matplotlib.figure.Figure):
    """
    Add debug artists to a figure that shows subfigure axis alignment.

    Parameters
    ----------
    fig: matplotlib Figure
        Figure that contains subfigures with axes to show debug info for.

    Returns
    -------
    None; modifies the figure in place.
    """
    n_subfigs = len(fig.subfigs)
    for idx, subfig in enumerate(fig.subfigs):
        figbox_color = matplotlib.colors.hsv_to_rgb(((idx / n_subfigs), 0.1, 1.0))
        fig.add_artist(
            matplotlib.patches.Rectangle(
                (0, 0), 1, 1, fc=figbox_color, ec=None, transform=subfig.transSubfigure, zorder=10
            )
        )

        guide_color = matplotlib.colors.hsv_to_rgb(((idx / n_subfigs), 0.6, 1.0))
        guide_kwargs = {"transform": subfig.transFigure, "zorder": 11, "color": guide_color}
        transform = subfig.transSubfigure + subfig.transFigure.inverted()

        for ax in subfig.axes:
            xmin, ymin, xmax, ymax = ax.get_position().extents
            xmin = transform.transform([xmin, 0.0])[0]
            xmax = transform.transform([xmax, 0.0])[0]
            ymin = transform.transform([0.0, ymin])[1]
            ymax = transform.transform([0.0, ymax])[1]
            fig.add_artist(matplotlib.lines.Line2D((xmin, xmin), (0.0, 1.0), **guide_kwargs))
            fig.add_artist(matplotlib.lines.Line2D((xmax, xmax), (0.0, 1.0), **guide_kwargs))
            fig.add_artist(matplotlib.lines.Line2D((0.0, 1.0), (ymin, ymin), **guide_kwargs))
            fig.add_artist(matplotlib.lines.Line2D((0.0, 1.0), (ymax, ymax), **guide_kwargs))

"""
Common function for analyzing flow data in Pandas Dataframes.

Allows users to specify custom metadata applied via well mapping.
Combines user data from multiple .csv files into a single DataFrame.
"""
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Support Python 3.7 by importing Literal from typing_extensions
try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit

from . import well_mapper


class MetadataWarning(UserWarning):
    """Warning raised when the passed metadata is possibly incorrect, but valid."""


class YamlError(RuntimeError):
    """Error raised when there is an issue with the provided .yaml file."""


class RegexError(RuntimeError):
    """Error raised when there is an issue with the file name regular expression."""


class GroupsError(RuntimeError):
    """Error raised when there is an issue with the data groups DataFrame."""


class MOIinputError(RuntimeError):
    """Error raised when there is an issue with the provided dataframe."""


def load_well_metadata(yaml_path: Union[str, Path]) -> Dict[Any, Any]:
    """Load a YAML file and convert it into a well mapping.

    Parameters
    ----------
    yaml_path: Path to the .yaml file to use for associating metadata with well IDs.

    Returns
    -------
    A dictionary that contains a well mapping for all metadata columns.
    """
    if not isinstance(yaml_path, Path):
        yaml_path = Path(yaml_path)

    with yaml_path.open() as yaml_file:
        metadata = yaml.safe_load(yaml_file)
        if (type(metadata) is not dict) or ("metadata" not in metadata):
            raise YamlError(
                "Incorrectly formatted .yaml file."
                " All metadata must be stored under the header 'metadata'"
            )
        for k, v in metadata["metadata"].items():
            if isinstance(v, dict):
                warnings.warn(
                    f'Metadata column "{k}" is a YAML dictionary, not a list!'
                    " Make sure your entries under this key start with dashes."
                    " Passing a dictionary does not allow duplicate keys and"
                    " is sort-order-dependent.",
                    MetadataWarning,
                    stacklevel=2,
                )
    return {k: well_mapper.well_mapping(v) for k, v in metadata["metadata"].items()}


def load_csv_with_metadata(
    data_path: Union[str, Path], yaml_path: Union[str, Path], filename_regex: Optional[str] = None
) -> pd.DataFrame:
    """
    Load .csv data into DataFrame with associated metadata.

    Generates a pandas DataFrame from a set of .csv files located at the given path,
    adding columns for metadata encoded by a given .yaml file. Metadata is associated
    with the data based on well IDs encoded in the data filenames.

    Parameters
    ----------
    data_path: str or Path
        Path to directory containing data files (.csv)
    yaml_path: str or Path
        Path to .yaml file to use for associating metadata with well IDs.
        All metadata must be contained under the header 'metadata'.
    filename_regex: str or raw str (optional)
        Regular expression to use to extract well IDs from data filenames.
        Must contain the capturing group 'well' for the sample well IDs.
        If not included, the filenames are assumed to follow this format (default
        export format from FlowJo): 'export_[well]_[population].csv'

    Returns
    -------
    A single pandas DataFrame containing all data with associated metadata.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    try:
        metadata_map = load_well_metadata(yaml_path)
    except FileNotFoundError as err:
        raise YamlError("Specified metadata YAML file does not exist!") from err

    # Load data from .csv files
    data_list: List[pd.DataFrame] = []

    for file in data_path.glob("*.csv"):

        # Default filename from FlowJo export is 'export_[well]_[population].csv'
        if filename_regex is None:
            filename_regex = r"^.*export_(?P<well>[A-P]\d+)_(?P<population>.+)\.csv"

        regex = re.compile(filename_regex)
        if "well" not in regex.groupindex:
            raise RegexError("Regular expression does not contain capturing group 'well'")
        match = regex.match(file.name)
        if match is None:
            continue

        # Load data
        df = pd.read_csv(file)

        # Add metadata to DataFrame
        well = match.group("well")
        index = 0
        for k, v in metadata_map.items():
            # Replace custom metadata keys with <NA> if not present
            df.insert(index, k, v[well] if well in v else [pd.NA] * len(df))
            index += 1

        for k in regex.groupindex.keys():
            df.insert(index, k, match.group(k))
            index += 1

        data_list.append(df)

    # Concatenate all the data into a single DataFrame
    if len(data_list) == 0:
        raise RegexError(f"No data files match the regular expression '{filename_regex}'")
    else:
        data = pd.concat(data_list, ignore_index=True).replace(np.NaN, pd.NA)  # type: ignore

    return data


def load_groups_with_metadata(
    groups_df: pd.DataFrame,
    base_path: Optional[Union[str, Path]] = "",
    filename_regex: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load .csv data into DataFrame with associated metadata by group.

    Each group of .csv files may be located at a different path and be
    associated with additional user-defined metadata.

    Parameters
    ----------
    groups_df: Pandas DataFrame
        Each row of the DataFrame is evaluated as a separate group. Columns must
        include 'data_path' and 'yaml_path', specifying absolute or relative paths
        to the group of .csv files and metadata .yaml files, respectively.
        Optionally, regular expressions for the file names can be specified for each
        group using the column 'filename_regex' (this will override the
        'filename_regex' argument).
    base_path: str or Path (optional)
        If specified, path that data and yaml paths in input_df are defined relative to.
    filename_regex: str or raw str (optional)
        Regular expression to use to extract well IDs from data filenames.
        Must contain the capturing group 'well' for the sample well IDs.
        Other capturing groups in the regex will be added as metadata.
        This value applies to all groups; to specify different regexes for each group,
        add the column 'filename_regex' to groups_df (this will override the
        'filename_regex' argument).
        If not included, the filenames are assumed to follow this format (default
        export format from FlowJo): 'export_[well]_[population].csv'

    Returns
    -------
    A single pandas DataFrame containing data from all groups with associated metadata.
    """
    if "data_path" not in groups_df.columns:
        raise GroupsError("'groups_df' must contain column 'data_path'")
    if "yaml_path" not in groups_df.columns:
        raise GroupsError("'groups_df' must contain column 'yaml_path'")

    if base_path and not isinstance(base_path, Path):
        base_path = Path(base_path)
    elif not base_path:
        base_path = ""

    group_list: List[pd.DataFrame] = []
    for group in groups_df.to_dict(orient="index").values():

        # Load data in group
        data_path = base_path / Path(group["data_path"])
        yaml_path = base_path / Path(group["yaml_path"])
        if "filename_regex" in groups_df.columns:
            filename_regex = group["filename_regex"]
        group_data = load_csv_with_metadata(data_path, yaml_path, filename_regex)

        # Add associated metadata (not paths)
        for k, v in group.items():
            if not (k == "data_path") and not (k == "yaml_path"):
                group_data[k] = v

        group_list.append(group_data)

    # Concatenate all the data into a single DataFrame
    data = pd.concat(group_list, ignore_index=True).replace(np.NaN, pd.NA)
    return data


def moi(
    data_frame: pd.DataFrame,
    color_column_name: str,
    color_cutoff: float,
    output_path: Optional[Union[str, Path]] = None,
    summary_method: Union[Literal["mean"], Literal["median"]] = "median",
) -> pd.DataFrame:
    """
    Calculate moi information from flowjo data with appropriate metadata.

    Generates a pandas DataFrame of virus titers from a pandas DataFrame of flowjo data.

    Parameters
    ----------
    data_frame: pd.DataFrame
        The pandas DataFrame to analyze. It must have the following columns:
            condition: the conditions/types of virus being analyzed
            replicate: the replicate of the data (can have all data as the same replicate)
            starting_cell_count: the number of cells in the well at the time of infection
            scaling: the dilution factor of each row
            max_virus: the maximum virus added to that column
                scaling times max_virus should result in the volume of virus stock added to a well
    color_column_name: str
        The name of the column on which to gate infection.
    color_cutoff: float
        The level of fluoresence on which to gate infecction.
    output_path: str or path (optional)
        The path to the output folder. If None, instead prints all plots to screen. Defaults to None
    summary_method: str (optional)
        Whether to return the calculated titer as the mean or median of the replicates.

    Returns
    -------
    A single pandas DataFrame containing the titer of each condition in TU per uL.
    """
    df = data_frame.copy()
    if color_column_name not in df.columns:
        raise MOIinputError(f"Input dataframe does not have a column called {color_column_name}")

    if output_path is not None:
        (Path(output_path) / "figures").mkdir(parents=True, exist_ok=True)

    if {"condition", "replicate", "starting_cell_count", "scaling", "max_virus"}.issubset(
        df.columns
    ):
        df["virus_amount"] = df["scaling"] * df["max_virus"]
        int_df = df[(df[color_column_name] > color_cutoff)]

        # Summarize cell counts for virus
        sum_df = (
            int_df.groupby(["condition", "replicate", "starting_cell_count", "virus_amount"])
            .count()
            .iloc[:, 0]
        )
        sum_df = sum_df.reset_index()
        sum_df.columns.values[4] = "virus_cell_count"
        # Summarize cell counts overall
        overall_counts = (
            df.groupby(["condition", "replicate", "starting_cell_count", "virus_amount"])
            .count()
            .iloc[:, 0]
        )
        overall_counts = overall_counts.reset_index()
        overall_counts.columns.values[4] = "flowed_cell_count"
        # Merge into one dataframe
        sum_df = pd.merge(
            sum_df,
            overall_counts,
            how="outer",
            on=["condition", "replicate", "starting_cell_count", "virus_amount"],
        )
        sum_df["virus_cell_count"] = sum_df["virus_cell_count"].fillna(0)

        # Calculate fraction infected, moi, and the titer
        sum_df["fraction_inf"] = sum_df["virus_cell_count"] / sum_df["flowed_cell_count"]

        def poisson_model(virus_vol, tui_ratio_per_vol):
            return 1 - np.exp(-tui_ratio_per_vol * virus_vol)

        # create the final dataframe
        final_titers = (
            sum_df.groupby(["condition", "replicate", "starting_cell_count"]).count().iloc[:, 0]
        )
        final_titers = final_titers.reset_index()
        final_titers.columns.values[3] = "tui_ratio_per_vol"

        tui = []
        # Calculate TU per cell per vol for each condition/replicate
        # via curvefit, then graph expected fraction infected for each uL of virus
        # and graph/save best fit
        for cond in np.unique(sum_df["condition"]):
            current_df = sum_df.loc[(sum_df["condition"] == cond)]
            plt.figure()
            for rep in np.unique(current_df["replicate"]):
                plot_df = current_df.loc[(current_df["replicate"] == rep)]
                plot_df = plot_df.sort_values("virus_amount")

                popt, _ = curve_fit(
                    poisson_model,
                    plot_df["virus_amount"],
                    plot_df["fraction_inf"],
                    p0=0.5,
                    bounds=(0, np.inf),
                )

                plt.scatter(plot_df["virus_amount"], plot_df["fraction_inf"])
                plt.plot(plot_df["virus_amount"], poisson_model(plot_df["virus_amount"], *popt))
                tui.append(popt[0])
            plt.title(f"Best Fit of Poisson Distribution for {cond}")
            plt.xscale("log")
            plt.ylabel("Fraction infected")
            plt.xlabel("Log (uL of virus in well)")
            if output_path is None:
                plt.show()
            else:
                plt.savefig(
                    Path(output_path) / "figures" / f"{str(cond)}_titer.png", bbox_inches="tight"
                )
            # graph MOI vs Fraction Infected with reference line
            plt.figure()
            plt.plot(np.linspace(0.0001, 2.3, 100), 1 - np.exp(-np.linspace(0.0001, 2.3, 100)))
            for rep in np.unique(current_df["replicate"]):
                plot_df = current_df[(current_df["replicate"] == rep)]
                plot_df = plot_df.sort_values("virus_amount")
                popt, _ = curve_fit(poisson_model, plot_df["virus_amount"], plot_df["fraction_inf"])
                plt.scatter(popt[0] * plot_df["virus_amount"], plot_df["fraction_inf"])
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Log MOI")
            plt.ylabel("Log Fraction Infected")
            plt.title(f"MOI v Fraction Infected Spread for {cond}")
            if output_path is None:
                plt.show()
            else:
                plt.savefig(
                    Path(output_path) / "figures" / f"{str(cond)}_MOIcurve.png", bbox_inches="tight"
                )
        # convert TU per cell per vol to TU per uL
        final_titers["moi"] = tui
        final_titers["titer_in_uL"] = final_titers["moi"] * final_titers["starting_cell_count"]
        if summary_method == "mean":
            final_output = final_titers.groupby("condition").mean()
        else:
            final_output = final_titers.groupby("condition").median()
        if output_path is not None:
            final_output.to_csv(Path(output_path) / "MOI_titer_data.csv")
        return final_output
    else:
        want = {"condition", "replicate", "starting_cell_count", "scaling", "max_virus"}
        have = df.columns
        lost = want.difference(have)
        raise MOIinputError(f"Missing the following columns from the input dataframe: {lost}")

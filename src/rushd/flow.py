"""
Common functions for analyzing flow data in Pandas Dataframes.

Allows users to specify custom metadata applied via well mapping.
Combines user data from multiple .csv files into a single DataFrame.
"""

import re
import warnings
from io import SEEK_SET, TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit

from . import well_mapper

OptionalZipPath = Union[str, Path, Tuple[Union[str, Path], str]]


class MetadataWarning(UserWarning):
    """Warning raised when the passed metadata is possibly incorrect, but valid."""


class YamlError(RuntimeError):
    """Error raised when there is an issue with the provided .yaml file."""


class RegexError(RuntimeError):
    """Error raised when there is an issue with the file name regular expression."""


class GroupsError(RuntimeError):
    """Error raised when there is an issue with the data groups DataFrame."""


class ColumnError(RuntimeError):
    """Error raised when the data is missing a column specifying well IDs."""


class DataPathError(RuntimeError):
    """Error raised when the path to the data is not specified correctly."""


class MOIinputError(RuntimeError):
    """Error raised when there is an issue with the provided dataframe."""


def _load_metadata_from_stream(stream: TextIOWrapper) -> Dict[Any, Any]:
    """
    Load YAML metadata from a stream-like object.

    Parameters
    ----------
    stream: a stream-like object to open as a YAML file

    Returns
    -------
    dict
        A dictionary that contains a well mapping for all metadata columns.
        Mapping is formatted as {key -> {well -> value}}.
    """
    metadata = yaml.safe_load(stream)
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


def load_well_metadata(yaml_path: OptionalZipPath) -> Dict[Any, Any]:
    """
    Load a YAML file and convert it into a well mapping.

    Parameters
    ----------
    yaml_path: str, Path, or a tuple of a str/Path zip file and a filename within
        Path to the .yaml file to use for associating metadata with well IDs.

    Returns
    -------
    dict
        A dictionary that contains a well mapping for all metadata columns.
        Mapping is formatted as {key -> {well -> value}}.
    """
    # load as entry in zip
    if isinstance(yaml_path, tuple):
        with ZipFile(yaml_path[0], "r") as zipfile:
            with TextIOWrapper(zipfile.open(yaml_path[1]), encoding="utf-8") as yaml_file:
                return _load_metadata_from_stream(yaml_file)

    # not a zip: load file directly
    if not isinstance(yaml_path, Path):
        yaml_path = Path(yaml_path)

    with yaml_path.open() as yaml_file:
        return _load_metadata_from_stream(yaml_file)


def _load_csv_from_stream(
    stream: TextIOWrapper,
    *,
    regex: re.Pattern,
    match: re.Match,
    csv_kwargs: Dict[str, Any],
    metadata_map: Optional[Dict[Any, Any]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV file from a stream and extra given metadata.

    Parameters
    ----------
    stream: TextIOWrapper
        A (rewindable) stream that stores CSV content
    regex: re.Pattern
        The regex used to generate the match
    match: re.Match
        The regex match object to use
    csv_kwargs: dict
        Additional kwargs to pass to pandas ``read_csv``. For instance, to skip rows or
        to specify alternate delimiters.
    metadata_map: dict
        The dictionary mapping
    columns: list of strings (optional)
        If specified, only these columns are loaded out of the .csv files.
        This can drastically reduce the amount of memory required to load
        flow data.

    """
    # Load the first row so we get the column names, then reset stream position
    stream_loc = stream.tell()
    df_onerow = pd.read_csv(stream, nrows=1, **csv_kwargs)
    stream.seek(stream_loc, SEEK_SET)

    # Load data: we allow extra columns in our column list, so subset it
    valid_cols = (
        list(set(columns).intersection(set(df_onerow.columns))) if columns is not None else None
    )
    df = pd.read_csv(stream, usecols=valid_cols, **csv_kwargs)

    # Add metadata to DataFrame
    index = 0
    # add YAML-extracted metadata
    if metadata_map is not None:
        well = match.group("well")
        for k, v in metadata_map.items():
            # Replace custom metadata keys with <NA> if not present
            df.insert(index, k, v[well] if well in v else [pd.NA] * len(df))
            index += 1

    # add filename metadata
    for k in regex.groupindex.keys():
        df.insert(index, k, match.group(k))
        index += 1
    return df


def load_csv_with_metadata(
    data_path: OptionalZipPath,
    yaml_path: OptionalZipPath,
    filename_regex: Optional[str] = None,
    *,
    columns: Optional[List[str]] = None,
    csv_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load .csv data into DataFrame with associated metadata.

    Generates a pandas DataFrame from a set of .csv files located at the given path,
    adding columns for metadata encoded by a given .yaml file. Metadata is associated
    with the data based on well IDs encoded in the data filenames.

    Parameters
    ----------
    data_path: location of the .csv files
        Either a directory containing .csv files,
        a zip file containing .csv files,
        or a path within a zip file containing .csv files.
    yaml_path: either a path to a .yaml file or a path within a zip file to a .yaml
        Path to .yaml file to use for associating metadata with well IDs.
        All metadata must be contained under the header 'metadata'.
    filename_regex: str or raw str (optional)
        Regular expression to use to extract well IDs from data filenames.
        Must contain the capturing group 'well' for the sample well IDs.
        If not included, the filenames are assumed to follow this format (default
        export format from FlowJo): ``export_[well]_[population].csv``
    columns: list of strings (optional)
        If specified, only these columns are loaded out of the .csv files.
        This can drastically reduce the amount of memory required to load
        flow data.
    csv_kwargs: dict (optional)
        Additional kwargs to pass to pandas ``read_csv``. For instance, to skip rows or
        to specify alternate delimiters.

    Returns
    -------
    DataFrame
        A single pandas DataFrame containing all data with associated metadata.
    """
    # normalize to Path objects
    if isinstance(data_path, tuple):
        normed_data_path: Union[Path, Tuple[Path, str]] = (Path(data_path[0]), data_path[1])
    else:
        normed_data_path: Union[Path, Tuple[Path, str]] = Path(data_path)

    try:
        metadata_map = load_well_metadata(yaml_path)
    except FileNotFoundError as err:
        raise YamlError("Specified metadata YAML file does not exist!") from err

    if csv_kwargs is None:
        csv_kwargs = {}

    # Load data from .csv files
    data_list: List[pd.DataFrame] = []

    # Default filename from FlowJo export is 'export_[well]_[population].csv'
    if filename_regex is None:
        filename_regex = r"^.*export_(?P<well>[A-P]\d+)_(?P<population>.+)\.csv"

    regex = re.compile(filename_regex)
    if "well" not in regex.groupindex:
        raise RegexError("Regular expression does not contain capturing group 'well'")

    if not isinstance(normed_data_path, tuple) and normed_data_path.is_dir():
        for file in normed_data_path.glob("*.csv"):
            match = regex.match(file.name)
            if match is None:
                continue

            with file.open("r") as csv_stream:
                df = _load_csv_from_stream(
                    csv_stream,
                    regex=regex,
                    match=match,
                    csv_kwargs=csv_kwargs,
                    metadata_map=metadata_map,
                    columns=columns,
                )
            data_list.append(df)
    else:
        # we need to open a zip file
        if isinstance(normed_data_path, tuple):
            zip_path = normed_data_path[0]
            rel_to = Path(normed_data_path[1])
        else:
            zip_path = normed_data_path
            rel_to = Path(".")
        # iterate over zip members, only selecting those that match the filename and are relative to
        # the correct directory
        with ZipFile(zip_path) as data_zip:
            for file in data_zip.infolist():
                filename = Path(file.filename)
                if filename.is_relative_to(rel_to):
                    match = regex.match(filename.name)
                    if match is None:
                        continue

                    with TextIOWrapper(
                        data_zip.open(file.filename), encoding="utf-8"
                    ) as csv_stream:
                        df = _load_csv_from_stream(
                            csv_stream,
                            regex=regex,
                            match=match,
                            csv_kwargs=csv_kwargs,
                            metadata_map=metadata_map,
                            columns=columns,
                        )
                    data_list.append(df)

    # Concatenate all the data into a single DataFrame
    if len(data_list) == 0:
        raise RegexError(f"No data files match the regular expression '{filename_regex}'")
    else:
        data = pd.concat(data_list, ignore_index=True).replace([float("nan"), np.nan], pd.NA)  # type: ignore

    return data


def load_groups_with_metadata(
    groups_df: pd.DataFrame,
    base_path: Optional[Union[str, Path]] = "",
    filename_regex: Optional[str] = None,
    *,
    columns: Optional[List[str]] = None,
    csv_kwargs: Optional[Dict[str, Any]] = None,
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
        `filename_regex` argument).
    base_path: str or Path (optional)
        If specified, path that data and yaml paths in input_df are defined relative to.
    filename_regex: str or raw str (optional)
        Regular expression to use to extract well IDs from data filenames.
        Must contain the capturing group `well` for the sample well IDs.
        Other capturing groups in the regex will be added as metadata.
        This value applies to all groups; to specify different regexes for each group,
        add the column 'filename_regex' to `groups_df` (this will override the
        `filename_regex` argument).
        If not included, the filenames are assumed to follow this format (default
        export format from FlowJo): ``export_[well]_[population].csv``
    columns: list of strings (optional)
        If specified, only these columns are loaded out of the .csv files.
        This can drastically reduce the amount of memory required to load
        flow data.
    csv_kwargs: dict (optional)
        Additional kwargs to pass to pandas ``read_csv``. For instance, to skip rows or
        to specify alternate delimiters.

    Returns
    -------
    DataFrame
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
        group_data_path: OptionalZipPath = group["data_path"]
        group_yaml_path: OptionalZipPath = group["yaml_path"]
        # Load data in group
        if isinstance(group_data_path, tuple):
            data_path = (base_path / Path(group_data_path[0]), group_data_path[1])
        else:
            data_path = base_path / Path(group_data_path)
        if isinstance(group_yaml_path, tuple):
            yaml_path = (base_path / Path(group_yaml_path[0]), group_yaml_path[1])
        else:
            yaml_path = base_path / Path(group_yaml_path)

        if "filename_regex" in groups_df.columns:
            filename_regex = group["filename_regex"]
        group_data = load_csv_with_metadata(
            data_path, yaml_path, filename_regex, columns=columns, csv_kwargs=csv_kwargs
        )

        # Add associated metadata (not paths)
        for k, v in group.items():
            if not (k == "data_path") and not (k == "yaml_path"):
                group_data[k] = v

        group_list.append(group_data)

    # Concatenate all the data into a single DataFrame
    data = pd.concat(group_list, ignore_index=True).replace([float("nan"), np.nan], pd.NA)
    return data


def load_csv(
    data_path: OptionalZipPath,
    filename_regex: Optional[str] = None,
    *,
    columns: Optional[List[str]] = None,
    csv_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load .csv data into DataFrame without additional metadata.

    Generates a pandas DataFrame from a set of .csv files located at the given path,
    adding columns for metadata encoded in the data filenames.

    Parameters
    ----------
    data_path: str or Path or a tuple with a str/Path to a zip file and a str folder inside
        Path to directory containing data files (.csv)
    filename_regex: str or raw str (optional)
        Regular expression to use to extract metadata from data filenames.
        Any named capturing groups will be added as metadata.
        If not included, the filenames are assumed to follow this format (default
        export format from FlowJo): ``export_[condition]_[population].csv``
    columns: list of strings (optional)
        If specified, only these columns are loaded out of the .csv files.
        This can drastically reduce the amount of memory required to load
        flow data.
    csv_kwargs: dict (optional)
        Additional kwargs to pass to pandas ``read_csv``. For instance, to skip rows or
        to specify alternate delimiters.

    Returns
    -------
    DataFrame
        A single pandas DataFrame containing all data with associated filename metadata.
    """
    # normalize to Path objects
    if isinstance(data_path, tuple):
        normed_data_path: Union[Path, Tuple[Path, str]] = (Path(data_path[0]), data_path[1])
    else:
        normed_data_path: Union[Path, Tuple[Path, str]] = Path(data_path)

    if csv_kwargs is None:
        csv_kwargs = {}

    # Load data from .csv files
    data_list: List[pd.DataFrame] = []

    # Default filename from FlowJo export is 'export_[sample name]_[population].csv'
    if filename_regex is None:
        filename_regex = r"^.*export_(?P<condition>[A-P]\d+)_(?P<population>.+)\.csv"

    regex = re.compile(filename_regex)

    if not isinstance(normed_data_path, tuple) and normed_data_path.is_dir():
        for file in normed_data_path.glob("*.csv"):
            match = regex.match(file.name)
            if match is None:
                continue

            with file.open("r") as csv_stream:
                df = _load_csv_from_stream(
                    csv_stream,
                    regex=regex,
                    match=match,
                    csv_kwargs=csv_kwargs,
                    columns=columns,
                )
            data_list.append(df)
    else:
        # we need to open a zip file
        if isinstance(normed_data_path, tuple):
            zip_path = normed_data_path[0]
            rel_to = Path(normed_data_path[1])
        else:
            zip_path = normed_data_path
            rel_to = Path(".")
        # iterate over zip members, only selecting those that match the filename and are relative to
        # the correct directory
        with ZipFile(zip_path) as data_zip:
            for file in data_zip.infolist():
                filename = Path(file.filename)
                if filename.is_relative_to(rel_to):
                    match = regex.match(filename.name)
                    if match is None:
                        continue

                    with TextIOWrapper(
                        data_zip.open(file.filename), encoding="utf-8"
                    ) as csv_stream:
                        df = _load_csv_from_stream(
                            csv_stream,
                            regex=regex,
                            match=match,
                            csv_kwargs=csv_kwargs,
                            columns=columns,
                        )
                    data_list.append(df)

    # Concatenate all the data into a single DataFrame
    if len(data_list) == 0:
        raise RegexError(f"No data files match the regular expression '{filename_regex}'")
    else:
        data = pd.concat(data_list, ignore_index=True).replace([float("nan"), np.nan], pd.NA)  # type: ignore

    return data


def moi(
    data_frame: pd.DataFrame,
    color_column_name: str,
    color_cutoff: float,
    output_path: Optional[Union[str, Path]] = None,
    summary_method: Union[Literal["mean"], Literal["median"]] = "median",
    *,
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate moi information from flowjo data with appropriate metadata.

    Generates a pandas DataFrame of virus titers from a pandas DataFrame of flowjo data.

    Parameters
    ----------
    data_frame: pd.DataFrame
        The pandas DataFrame to analyze. It must have the following columns:
            - condition: the conditions/types of virus being analyzed
            - replicate: the replicate of the data (can have all data as the same replicate)
            - starting_cell_count: the number of cells in the well at the time of infection
            - scaling: the dilution factor of each row
            - max_virus: the maximum virus added to that column
              scaling times max_virus should result in the volume of virus stock added to a well
    color_column_name: str
        The name of the column on which to gate infection.
    color_cutoff: float
        The level of fluoresence on which to gate infecction.
    output_path: str or path (optional)
        The path to the output folder. If None, instead prints all plots to screen. Defaults to None
    summary_method: str (optional)
        Whether to return the calculated titer as the mean or median of the replicates.
    scale_factor: float (optional)
        Whether to scale down the Poisson fit by the given scale factor maximum.

    Returns
    -------
    DataFrame
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
            return scale_factor * (1 - np.exp(-tui_ratio_per_vol * virus_vol))

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
                plt.scatter(
                    scale_factor * popt[0] * plot_df["virus_amount"], plot_df["fraction_inf"]
                )
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

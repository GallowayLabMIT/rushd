"""
Common functions for analyzing qPCR data in Pandas Dataframes.

Allows users to specify custom metadata applied via well mapping.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import flow


class YamlError(RuntimeError):
    """Error raised when there is an issue with the provided .yaml file."""


class ColumnError(RuntimeError):
    """Error raised when the data is missing a required column."""


class DataPathError(RuntimeError):
    """Error raised when the path to the data is not specified correctly."""


class GroupsError(RuntimeError):
    """Error raised when there is an issue with the data groups DataFrame."""


class RegexError(RuntimeError):
    """Error raised when there is an issue with the file name regular expression."""


class InputError(RuntimeError):
    """Error raised when there is an issue with an argument type."""


def load_single_csv_with_metadata(
    data_path: Union[str, Path],
    yaml_path: Union[str, Path],
    *,
    well_column: Optional[str] = "well",
    columns: Optional[List[str]] = None,
    csv_kwargs: Optional[Dict[str, Any]] = {},
    is_default: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Load .csv data into DataFrame with associated metadata.

    Generates a pandas DataFrame from a single .csv file located at the given path,
    adding columns for metadata encoded by a given .yaml file. Metadata is associated
    with the data based on well IDs encoded in one of the data columns.

    Note that this uses pandas 'read_csv', so it is compatible with .tsv and .txt files
    with the appropriate kwargs.

    Parameters
    ----------
    data_path: str or Path
        Path to directory containing data files (.csv or similar)
    yaml_path: str or Path
        Path to .yaml file to use for associating metadata with well IDs.
        All metadata must be contained under the header 'metadata'.
    well_column: str, default 'well'
        Name of the column containing well IDs.
    columns: List of strings (optional)
        If specified, only these columns are loaded out of the .csv files.
        This can drastically reduce the amount of memory required to load
        flow data.
    csv_kwargs: dict (optional)
        Additional kwargs to pass to pandas 'read_csv'. For instance, to skip rows or
        to specify alternate delimiters.
    is_default: bool, default False
        If True, will override 'well_column', 'columns', and 'csv_kwargs' with
        defaults for plates with the format exported from Roche LightCycler 480II.

    Returns
    -------
    A single pandas DataFrame containing all data with associated metadata.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    try:
        metadata_map = flow.load_well_metadata(yaml_path)
    except FileNotFoundError as err:
        raise YamlError("Specified metadata YAML file does not exist!") from err

    # Check that a single file (not a directory) has been passed
    if data_path.is_dir():
        raise DataPathError(
            "'data_path' must be a single file. To load multiple files, use 'load_csv_with_metadata'"
        )
    file = data_path

    # Overwrite args with those relevant for
    if is_default:
        well_column = "Pos"
        columns = ["Cp"]
        csv_kwargs = dict(sep="\t", header=1)

    # Load the first row so we get the column names
    df_onerow = pd.read_csv(file, nrows=1, **csv_kwargs)
    # Load data: we allow extra columns in our column list, so subset it
    valid_cols = (
        list(set(columns + [well_column]).intersection(set(df_onerow.columns)))
        if columns is not None
        else None
    )
    data = pd.read_csv(file, usecols=valid_cols, **csv_kwargs)

    if well_column not in data.columns:
        raise (ColumnError(f"The file at 'data_path' does not contain the column '{well_column}'"))

    # Add metadata to DataFrame
    metadata = pd.DataFrame.from_dict(metadata_map).reset_index(names="well")
    data = data.merge(metadata, how="left", left_on=well_column, right_on="well").replace(
        [float("nan"), np.nan], pd.NA
    )

    return data


def load_plates_with_metadata(
    groups_df: pd.DataFrame,
    base_path: Optional[Union[str, Path]] = "",
    filename_regex: Optional[str] = None,
    *,
    well_column: Optional[str] = "well",
    columns: Optional[List[str]] = None,
    csv_kwargs: Optional[Dict[str, Any]] = {},
    is_default: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Load data from multiple plates into a DataFrame with associated metadata.

    Each plate is a .csv file with well IDs encoded in one of the data columns.

    Parameters
    ----------
    groups_df: Pandas DataFrame
        Each row of the DataFrame is evaluated as a separate plate. Columns must
        include 'data_path' and 'yaml_path', specifying absolute or relative paths
        to the .csv files and metadata .yaml files, respectively.
        Optionally, regular expressions for the file names can be specified for each
        file using the column 'filename_regex' (this will override the
        'filename_regex' argument).
    base_path: str or Path (optional)
        If specified, path that data and yaml paths in input_df are defined relative to.
    filename_regex: str or raw str (optional)
        Regular expression to use to extract metadata from data filenames.
        This value applies to all groups; to specify different regexes for each group,
        add the column 'filename_regex' to groups_df (this will override the
        'filename_regex' argument).
        If not included, filename information will not be added as metadata.
    well_column: str, default 'well'
        Name of the column containing well IDs.
    columns: List of strings (optional)
        If specified, only these columns are loaded out of the .csv files.
        This can drastically reduce the amount of memory required to load
        flow data.
    csv_kwargs: dict (optional)
        Additional kwargs to pass to pandas 'read_csv'. For instance, to skip rows or
        to specify alternate delimiters.
    is_default: bool, default False
        If True, will override 'well_column', 'columns', and 'csv_kwargs' with
        defaults for plates with the format exported from Roche LightCycler 480II.

    Returns
    -------
    A single pandas DataFrame containing data from all plates with associated metadata.
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

        if filename_regex is not None:
            regex = re.compile(filename_regex)
            match = regex.match(data_path.name)
            if match is None:
                raise RegexError(
                    f"Filename does not match the regular expression '{filename_regex}'"
                )

        group_data = load_single_csv_with_metadata(
            data_path,
            yaml_path,
            well_column=well_column,
            columns=columns,
            csv_kwargs=csv_kwargs,
            is_default=is_default,
        )

        # Add associated metadata (not paths)
        index = 0
        for k, v in group.items():
            if not (k == "data_path") and not (k == "yaml_path"):
                group_data.insert(index, k, v)
                index += 1

        if filename_regex is not None:
            for k in regex.groupindex.keys():
                group_data.insert(index, k, match.group(k))
                index += 1

        group_list.append(group_data)

    # Concatenate all the data into a single DataFrame
    data = pd.concat(group_list, ignore_index=True).replace([float("nan"), np.nan], pd.NA)
    return data

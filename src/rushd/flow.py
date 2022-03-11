"""
Common function for analyzing flow data in Pandas Dataframes.

Allows users to specify custom metadata applied via well mapping.
Combines user data from multiple .csv files into a single DataFrame.
"""
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from . import well_mapper


class YamlError(RuntimeError):
    """Error raised when there is an issue with the provided .yaml file."""


class RegexError(RuntimeError):
    """Error raised when there is an issue with the file name regular expression."""


def load_csv_with_metadata(
    data_path: str, yaml_path: str, filename_regex: Optional[str] = None
) -> pd.DataFrame:
    """
    Load .csv data into DataFrame with associate metadata.

    Generates a pandas DataFrame from a set of .csv files located at the given path,
    adding columns for metadata encoded by a given .yaml file. Metadata is associated
    with the data based on well IDs encoded in the data filenames.

    Parameters
    ----------
    data_path: str
        Path to directory containing data files (.csv)
    yaml_path: str
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
    # Check if path to .yaml file is valid
    if yaml_path[-5:] != ".yaml":
        raise YamlError(f"{yaml_path} is not a .yaml file")

    with open(yaml_path) as file:
        metadata = yaml.safe_load(file)
        if (type(metadata) is not dict) or ("metadata" not in metadata):
            raise YamlError(
                "Incorrectly formatted .yaml file."
                " All metadata must be stored under the header 'metadata'"
            )
        metadata_map = {k: well_mapper.well_mapping(v) for k, v in metadata["metadata"].items()}

    # Load data from .csv files
    data_list: List[pd.DataFrame] = []

    for file in Path(data_path).glob("*.csv"):

        # Default filename from FlowJo export is 'export_[well]_[population].csv'
        if filename_regex is None:
            filename_regex = r"^.*export_(?P<well>[A-G0-9]+)_(?P<population>.+)\.csv"

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
            df.insert(index, k, v[well])
            index += 1

        for k in regex.groupindex.keys():
            df.insert(index, k, match.group(k))
            index += 1

        data_list.append(df)

    # Concatenate all the data into a single DataFrame
    if len(data_list) == 0:
        raise RegexError(f'No data files match the regular expression "{filename_regex}"')
    else:
        data = pd.concat(data_list, ignore_index=True)

    return data

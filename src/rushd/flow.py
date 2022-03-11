import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from . import well_mapper


def load_csv_with_metadata(
    data_path: str, yaml_path: str, filename_regex: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a pandas DataFrame from a set of .csv files located at the given path,
    adding columns for metadata encoded by a given .yaml file. Metadata is associated
    with the data based on well IDs encoded in the data filenames.

    Parameters
    ----------
    data_path: str
        Path to directory containing data files (.csv)
    yaml_path: str
        Path to directory containing .yaml file, or path to file itself,
        to use for associating metadata with well IDs
    filename_regex: str or raw str (optional)
        Regular expression to use to extract well IDs from data filenames.
        If not included, the filenames are assumed to follow this format (default
        export format from FlowJo): 'export_[well]_[other text].csv'

    Returns
    -------
    A single pandas DataFrame containing all data with associated metadata.
    """

    # Create well mapping from .yaml file

    # If the path is the actual yaml file and not just the directory that it's in, read that in
    # otherwise, read in the first yaml file found in the specified directory.
    if yaml_path[-5:] == ".yaml":
        f = yaml_path
    else:

        try:
            f = next(yaml_path.glob("*.yaml"))

        except StopIteration:
            print("No YAML file found in {}".format(yaml_path), file=sys.stderr)
            sys.exit(1)

    with open(f) as file:
        metadata = yaml.safe_load(file)
        metadata_map = {
            k: well_mapper.well_mapping(v) for k, v in metadata["metadata"].items()
        }

    # Load data from .csv files

    data_list: List[pd.DataFrame] = []

    for file in Path(data_path).glob("*.csv"):

        # Find data files

        if filename_regex is not None:
            regex = re.compile(filename_regex)

        else:
            # Default filename from FlowJo export is 'export_[well]_[population].csv'
            # Custom regex must contain capture group 'well'
            # TO DO: add extra categories as metadata
            regex = re.compile(r"^.*export_(?P<well>[A-G0-9]+)_(?P<population>.+)\.csv")

        match = regex.match(file.name)
        if match is None:
            continue

        # Load data
        df = pd.read_csv(file)

        # Add metadata to DataFrame

        well = match.group("well")
        # Fix well ID to format letter-digit-digit
        if len(well) == 2:
            well = well[:1] + "0" + well[1:]

        index = 0
        for k, v in metadata_map.items():
            df.insert(index, k, v[well])
            index += 1
        # Also add column for well ID
        df.insert(index, "Well", well)

        data_list.append(df)

    # Concatenate all the data into a single DataFrame
    data = pd.concat(data_list, ignore_index=True)

    return data

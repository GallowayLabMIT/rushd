"""
Common functions for analyzing ddPCR data in Pandas Dataframes.

Extracts data and metadata from .ddpcr files.
Allows users to specify custom metadata applied via well mapping.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Support Python 3.7 by importing Literal from typing_extensions
try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal

import json
import numpy as np
import pandas as pd
import py7zr
import shutil
import tempfile
import yaml

from . import well_mapper, flow


class YamlError(RuntimeError):
    """Error raised when there is an issue with the provided .yaml file."""

class DataPathError(RuntimeError):
    """Error raised when the path to the data is not specified correctly."""


def load_ddpcr_metadata(unzipped_path: Path) -> Dict[Any, Any]:
    """
    Load well metadata from an unzipped .ddpcr file.

    Generates a metadata dict in the same format as the YAML well mapping,
    i.e., key -> {well -> value}. The columns are a subset of the 
    metadata associated with each well in the BioRad software, namely 
    sample names (numbered 'Sample description' fields, returned as 
    numbered 'sample_description' keys) and targets for each channel/dye 
    (returned as '[channel]_target' keys).

    Parameters
    ----------
    unzipped_path: Path
        Path to unzipped .ddpcr file

    Returns
    -------
    A dictionary that contains a well mapping for metadata extracted from
    the .ddpcr experiment.
    """

    filename_regex = r"^.*[\\/](?P<well>[A-P]\d+)\.dd.*json"
        
    # Create map of well index -> ID 
    well_id_map = {}
    for f in (unzipped_path/'PeakMetaData').glob("*.ddmetajson"):
        with open(f, 'r') as file:
            d = json.load(file)
            well_id_map[d['WellIndex']] = re.compile(filename_regex).match(file.name).group("well")
    
    # Get plate file name from last modified .ddplt file
    plate_file = ''
    last_mod_time = 0
    for f in unzipped_path.glob("*.ddplt"):
        mtime = f.stat().st_mtime
        if mtime > last_mod_time:
            last_mod_time = mtime
            plate_file = f.name
    
    # Load metadata from plate file
    metadata_from_plt = {}
    with open(unzipped_path/plate_file, 'r') as file:
        f = json.load(file)
        for w in f['WellSamples']:
            well = well_id_map[w['WellIndex']]
            condition_map = {f'sample_description_{i+1}': val for i,val in enumerate(w['SampleIds'])}
            target_map = {p['Dye']['DyeName']+'_target': p['TargetName'] for p in w['Panel']['Targets']}
            metadata_from_plt[well] = condition_map | target_map
    
    metadata_map = pd.DataFrame.from_dict(metadata_from_plt, orient='index').to_dict()
    return metadata_map


def load_ddpcr(
    data_path: Union[str, Path],
    yaml_path: Optional[Union[str, Path]] = None,
    *,
    extract_metadata: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Load ddPCR data into DataFrame with associated metadata.

    Generates a pandas DataFrame from a .ddpcr file, which is the
    file type for experiments on the BioRad QX100/QX200 machines.
    Adds columns for metadata encoded by a given .yaml file. 
    Metadata is associated with the data based on well IDs extracted
    from the experiment data.

    Parameters
    ----------
    data_path: str or Path
        Path to .ddpcr file.
    yaml_path: str or Path (optional)
        Path to .yaml file to use for associating metadata with well IDs.
        All metadata must be contained under the header 'metadata'.
    extract_metadata: bool, default True
        Whether to extract metadata from the .ddpcr file. If True,
        adds a subset of the metadata associated with each well in the 
        BioRad software, namely sample names (numbered 'Sample description' fields,
        returned as numbered 'condition' keys) and targets for each channel/dye 
        (returned as '[channel]_target' keys).

    Returns
    -------
    A single pandas DataFrame containing all data with associated metadata.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    if data_path.suffix != '.ddpcr':
        raise DataPathError("'data_path' must be a .ddpcr file.")
    
    # Unzip .ddpcr file
    tmp_data_path = Path(tempfile.mkdtemp())
    with py7zr.SevenZipFile(data_path, 'r', password='1b53402e-503a-4303-bf86-71af1f3178dd') as experiment:
        experiment.extractall(path=tmp_data_path)

    metadata_map = {}

    # Load metadata from .yaml file
    if yaml_path is not None:
        try:
            metadata_map = flow.load_well_metadata(yaml_path)
        except FileNotFoundError as err:
            raise YamlError("Specified metadata YAML file does not exist!") from err
    
    # Load metadata from .ddpcr file
    if extract_metadata:
        metadata_map = metadata_map | load_ddpcr_metadata(tmp_data_path)

    # Load data for each well
    data_list = []
    for f in (tmp_data_path/'PeakData').glob("*.ddpeakjson"):
        with open(f, 'r') as file:
            d = json.load(file)

            # Ignore wells for which no data was collected
            if not d["DataAcquisitionInfo"]['WasAcquired']: continue

            # Extract raw data (channel amplitude) and channel names
            channel_map = {c['Channel']-1: c['Dye'] for c in d["DataAcquisitionInfo"]['ChannelMap']}
            df = pd.DataFrame(np.transpose(d['PeakInfo']['Amplitudes'])).rename(columns=channel_map)

            well = f.stem
            df.insert(0, 'well', [well]*len(df))

            # Add metadata to DataFrame
            index = 0
            for k, v in metadata_map.items():
                # Replace custom metadata keys with <NA> if not present
                df.insert(index, k, v[well] if well in v else [pd.NA] * len(df))
                index += 1

            data_list.append(df)

    # Fill empty values with <NA> and drop empty columns
    data = pd.concat(data_list, ignore_index=True).replace([float('nan'), np.nan, ''], pd.NA).dropna(axis='columns', how='all')

    # Delete unzipped files
    shutil.rmtree(tmp_data_path)
    
    return data


def calculate_copy_number(
    df: pd.DataFrame,
    exp_channel: str,
    ref_channel: str,
    gates: Dict[str, float],
    *,
    ref_copy_num: float = 2.0,
) -> pd.DataFrame:
    """
    Calculates copy number of an experimental target relative to a
    reference target.

    Adds a column to the DataFrame with this computed value.
    Math is based on ... TODO

    Parameters
    ----------
    df: pandas DataFrame
        Data on which to calculate. Must contain columns corresponding 
        to the experimental and reference channels.
    exp_channel: str
        Column in df containing measurements for the experimental target.
    ref_channel: str
        Column in df containing measurements for the reference target.
    gates: dict of (str: float) pairs
        Gates specifying threshold for positive droplets, one for each
        experimental and reference channel.
    ref_copy_num: float, default 2.0
        Known copy number of the reference gene. If not specified, defaults
        to 2.0 (diploid).

    Returns
    -------
    The original DataFrame with a new column 'copy_num' containing the computed
    values.
    """
    # TODO: throw error if channels not in df
    # TODO: check if there are no (negative) droplets before calculating
    data_exp = df[exp_channel]
    data_ref = df[ref_channel]

    # Compute copies per droplet: -ln(num_negative / num_total)
    copies_exp = -np.log((data_exp < gates[exp_channel]).sum() / len(data_exp))
    copies_ref = -np.log((data_ref < gates[ref_channel]).sum() / len(data_ref))

    # Normalize copies to the reference gene
    df['copy_num'] = copies_exp / copies_ref * ref_copy_num
    return df
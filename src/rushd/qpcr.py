"""
Common function for analyzing qPCR data in Pandas Dataframes.

Allows users to specify custom metadata applied via well mapping.
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

from . import flow

def load_qpcr_with_metadata(
    data_path: Union[str, Path],
    yaml_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Load qPCR data into DataFrame with associated metadata.

    Wrapper for 'load_single_csv_with_metadata' using default file format for qPCR data
    ('cp_table.txt' exported from Roche LightCycler 480II).

    Generates a pandas DataFrame from a single .csv file located at the given path,
    adding columns for metadata encoded by a given .yaml file. Metadata is associated
    with the data based on well IDs encoded in one of the data columns.

    Parameters
    ----------
    data_path: str or Path
        Path to single data file, in any file format accepted by pd.read_csv (e.g., .csv, .txt)
    yaml_path: str or Path
        Path to .yaml file to use for associating metadata with well IDs.
        All metadata must be contained under the header 'metadata'.

    Returns
    -------
    A single pandas DataFrame containing all data (Cp values) with metadata associated with each well.
    """
    return flow.load_single_csv_with_metadata(
        data_path,
        yaml_path,
        well_column='Pos',
        columns=['Cp'],
        csv_kwargs=dict(sep='\t', header=1)
        )

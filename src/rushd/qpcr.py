"""
Common functions for analyzing qPCR data in Pandas Dataframes.

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

import matplotlib
import numpy as np
import pandas as pd
import yaml
import scipy.stats
from scipy.optimize import curve_fit

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
    well_column: Optional[str] = 'well',
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
        raise DataPathError("'data_path' must be a single file. To load multiple files, use 'load_csv_with_metadata'")
    file = data_path

    # Overwrite args with those relevant for 
    if is_default:
        well_column = 'Pos'
        columns = ['Cp']
        csv_kwargs = dict(sep='\t', header=1)

    # Load the first row so we get the column names
    df_onerow = pd.read_csv(file, nrows=1, **csv_kwargs)
    # Load data: we allow extra columns in our column list, so subset it
    valid_cols = (
        list(set(columns+[well_column]).intersection(set(df_onerow.columns))) if columns is not None else None
    )
    data = pd.read_csv(file, usecols=valid_cols, **csv_kwargs)
        
    if well_column not in data.columns:
        raise(ColumnError(f"The file at 'data_path' does not contain the column '{well_column}'"))

    # Add metadata to DataFrame
    metadata = pd.DataFrame.from_dict(metadata_map).reset_index(names='well')
    data = data.merge(metadata, how='left', left_on=well_column, right_on='well').replace([float('nan'),np.nan], pd.NA)

    return data


def load_plates_with_metadata(
    groups_df: pd.DataFrame,
    base_path: Optional[Union[str, Path]] = "",
    filename_regex: Optional[str] = None,
    *,
    well_column: Optional[str] = 'well',
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
                raise RegexError(f"Filename does not match the regular expression '{filename_regex}'")

        group_data = load_single_csv_with_metadata(
            data_path, 
            yaml_path, 
            well_column=well_column, 
            columns=columns, 
            csv_kwargs=csv_kwargs, 
            is_default=is_default
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
    data = pd.concat(group_list, ignore_index=True).replace([float('nan'),np.nan], pd.NA)
    return data


def calculate_standard(
    df: pd.DataFrame,
    amt_col: str,
    cp_col: str,
    ax: Optional[matplotlib.axes] = None
) -> List[scipy.stats._stats_py.LinregressResult, float]:
    """
    Calculate a standard curve for qPCR data.

    For the given data, treats the values in 'amt_col' as 
    input amounts and values in 'cp_col' as the corresponding
    cycle counts (Cp, aka Ct) from the qPCR output. Computes a linear 
    regression on log10(amount) vs Cp, and returns this fit as well
    as the efficiency.

    If axes are passed, plots the linear fit on the data, annotating 
    the R^2 value and efficiency.

    Parameters
    ----------
    df: pandas DataFrame
        Data to use to fit.
    amt_col: str
        Name of column containing input amounts.
    cp_col: str
        Name of the column containing Cp values.
    ax: matplotlib.axes (optional)
        Axes on which to plot the data and fit.

    Returns
    -------
    A tuple of the fit (output of a call to scipy.stats.linregress)
    and the calculated efficiency (float).
    """
    if amt_col not in df.columns:
        raise ColumnError(f"Data is missing the 'amt_col' column {amt_col}")
    if cp_col not in df.columns:
        raise ColumnError(f"Data is missing the 'cp_col' column {cp_col}")
    
    # Remove zero values and log10-transform input amounts
    df_subset = df[df[amt_col]>0].copy()
    df_subset['log10_'+amt_col] = df_subset[amt_col].astype(float).apply(np.log10)

    # Fit data
    x = df_subset['log10_'+amt_col]
    y = df_subset[cp_col].astype(float)
    fit = scipy.stats.linregress(x, y)
    efficiency = (10**(-1/fit.slope) - 1)*100 # percentage
    
    # Plot result
    if ax is not None:
        ax.scatter(df[amt_col], df[cp_col], label='data', ec='white', lw=0.75)
        xs = np.logspace(min(df_subset['log10_'+amt_col]), max(df_subset['log10_'+amt_col]), 1000)
        ys = fit.slope * np.log10(xs) + fit.intercept
        ax.plot(xs, ys, color='crimson', label='linear\nregression')
        ax.set_xscale('symlog', linthresh=min(df_subset[amt_col]))
        pad = 0.01
        ax.legend(loc='upper right', bbox_to_anchor=(1-pad, 1-pad))
        ax.annotate(f'$R^2$: {abs(fit.rvalue):0.3f}', (0+pad*2, 0.1), xycoords='axes fraction',
                    ha='left', va='bottom', size='medium')
        ax.annotate(f'Efficiency: {efficiency:0.1f}%', (0+pad*2, 0+pad*2), xycoords='axes fraction',
                    ha='left', va='bottom', size='medium')
    
    return fit, efficiency


def calculate_input_amount(
    y: float, # TODO: list of float
    fit: Union[scipy.stats._stats_py.LinregressResult, List[float]],
) -> float:
    """
    Given a cycle count (Cp, aka Ct value) and a linear regression fit, 
    compute the amount of input.

    Note that the linear regression fit is expected to have been performed
    on the log10-transform of the input amounts. Units of the returned value
    match those of the non-transformed input amount data.

    Parameters
    ----------
    y: float
        Cycle count (Cp, aka Ct value).
    fit: scipy LinregressResult object or list of two floats
        Linear fit to use. Accepts either the output of a call
        to scipy.stats.linregress or a list of the fit values
        [slope, intercept].

    Returns
    -------
    A float of the calculated amount.
    """
    if type(fit) is scipy.stats._stats_py.LinregressResult:
        return 10**((float(y)-fit.intercept)/fit.slope)
    if len(fit) < 2:
        raise InputError("'fit' is expected to be a list containing [slope, intercept]. Alternatively, pass a scipy LinregressResult object.")
    return 10**((float(y)-fit[1])/fit[0])

# TODO: add 'type' arg for dsDNA, ssDNA, ssRNA
def convert_moles_to_mass(
    moles: Union[float, List[float]],
    length: Union[float, List[float]]
) -> Union[float, List[float]]:
    """
    For a given amount of DNA in moles, use its length
    to calculate its mass.

    Formula from NEB: 
    g = mol x (bp x 615.94 g/mol/bp + 36.04 g/mol)
     - mass of dsDNA (g) = moles dsDNA x (molecular weight of dsDNA (g/mol))
     - molecular weight of dsDNA = (number of base pairs of dsDNA x average molecular weight of a base pair) + 36.04 g/mol
     - average molecular weight of a base pair = 615.94 g/mol, excluding the water molecule removed during polymerization 
       and assuming deprotonated phosphate hydroxyls
     - the additional 36.04 g/mol accounts for the 2 -OH and 2 -H added back to the ends
     - bases are assumed to be unmodified

    Parameters
    ----------
    moles: float or list of float
        Amount of dsDNA in moles.
    length: float or list of float
        Number of base pairs of the dsDNA (or average length of a heterogeneous sample).
    
    Returns
    -------
    A float or list of floats of the calculated mass in grams.
    """
    return np.array(moles) * (np.array(length) * 615.96 + 36.04)


# TODO: add 'type' arg for dsDNA, ssDNA, ssRNA
def convert_mass_to_moles(
    mass: Union[float, List[float]],
    length: Union[float, List[float]]
) -> Union[float, List[float]]:
    """
    For a given amount of DNA in moles, use its length
    to calculate its mass.

    Formula from NEB: 
    mol = g / (bp x 615.94 g/mol/bp + 36.04 g/mol)
     - moles dsDNA = mass of dsDNA (g) / (molecular weight of dsDNA (g/mol))
     - molecular weight of dsDNA = (number of base pairs of dsDNA x average molecular weight of a base pair) + 36.04 g/mol
     - average molecular weight of a base pair = 615.94 g/mol, excluding the water molecule removed during polymerization 
       and assuming deprotonated phosphate hydroxyls
     - the additional 36.04 g/mol accounts for the 2 -OH and 2 -H added back to the ends
     - bases are assumed to be unmodified

    Parameters
    ----------
    mass: float or list of float
        Mass of dsDNA in grams.
    length: float or list of float
        Number of base pairs of the dsDNA (or average length of a heterogeneous sample).
    
    Returns
    -------
    A float or list of floats of the calculated amount in moles.
    """
    return np.array(mass) / (np.array(length) * 615.96 + 36.04)
import os
from pathlib import Path

import pandas as pd
import pytest

from rushd import ddpcr


class YamlError(RuntimeError):
    """Error raised when there is an issue with the provided .yaml file."""

class DataPathError(RuntimeError):
    """Error raised when the path to the data is not specified correctly."""


def test_ddpcr_examples():
    """
    Tests that several sample .ddpcr files can be loaded with the 
    expected channels and extracted metadata
    """
    df = ddpcr.load_ddpcr('tests/sample_data/2025.07.18-ddPCR.TANGLEs_20250718_123154_646.ddpcr')
    assert all(c in df.columns for c in ['well','HEX','FAM','HEX_target','FAM_target','sample_description_1'])

    df = ddpcr.load_ddpcr('tests/sample_data/ABA_ddPCR_2_20251112_20251112_151023_656.ddpcr')
    assert all(c in df.columns for c in ['well','HEX','FAM','HEX_target','FAM_target','sample_description_1'])

    df = ddpcr.load_ddpcr('tests/sample_data/Rogi2-Clybl-v3-monoclones-5-24_20241204_100204_335.ddpcr')
    assert all(c in df.columns for c in ['well','HEX','FAM','HEX_target','FAM_target','sample_description_1'])


def test_no_metadata():
    """
    Tests that .ddpcr files can be loaded without extracting metadata
    """
    df = ddpcr.load_ddpcr('tests/sample_data/2025.07.18-ddPCR.TANGLEs_20250718_123154_646.ddpcr', extract_metadata=False)
    assert all(c in df.columns for c in ['well','HEX','FAM'])
    assert all(c not in df.columns for c in ['HEX_target','FAM_target','sample_description_1'])

    df = ddpcr.load_ddpcr('tests/sample_data/ABA_ddPCR_2_20251112_20251112_151023_656.ddpcr', extract_metadata=False)
    assert all(c in df.columns for c in ['well','HEX','FAM'])
    assert all(c not in df.columns for c in ['HEX_target','FAM_target','sample_description_1'])


def test_yaml():
    """
    Tests that .ddpcr file can be loaded with metadata from associated .yaml file
    """
    df = ddpcr.load_ddpcr('tests/sample_data/2025.07.18-ddPCR.TANGLEs_20250718_123154_646.ddpcr',
                          'tests/sample_data/2025.07.18-ddPCR.TANGLEs_wells.yaml')
    assert all(c in df.columns for c in ['well','HEX','FAM_target','sample_description_1','cell_line','fam_target'])

    df = ddpcr.load_ddpcr('tests/sample_data/2025.07.18-ddPCR.TANGLEs_20250718_123154_646.ddpcr',
                          'tests/sample_data/2025.07.18-ddPCR.TANGLEs_wells.yaml', extract_metadata=False)
    assert all(c in df.columns for c in ['well','HEX','cell_line','fam_target'])


def test_invalid_data_path(tmp_path: Path):
    """
    Tests that proper error is thrown when data_path is not a .ddpcr file
    """
    with open(str(tmp_path / "bad_file_type.csv"), "w") as f:
        f.write("""well,HEX,FAM\nA1,10,20""")
    with pytest.raises(ddpcr.DataPathError): 
        _ = ddpcr.load_ddpcr(tmp_path/'bad_file_type.csv')


def test_invalid_yaml(tmp_path: Path):
    """
    Tests that invalid .yaml files throw errors
    """
    with pytest.raises(ddpcr.YamlError):
        _ = ddpcr.load_ddpcr('tests/sample_data/2025.07.18-ddPCR.TANGLEs_20250718_123154_646.ddpcr', tmp_path / "nonexistent.yaml")
    with pytest.raises(ddpcr.YamlError):
        _ = ddpcr.load_ddpcr('tests/sample_data/2025.07.18-ddPCR.TANGLEs_20250718_123154_646.ddpcr', "wells")


def test_multiple_plates():
    """
    Tests that the correct plate is selected when multiple .ddplt files exist
    """
    df = ddpcr.load_ddpcr('tests/sample_data/ABA_ddPCR_2_20251112_20251112_151023_656.ddpcr')
    assert 'EBFP' in df['FAM_target'].unique() # this is not labeled in other plate
    assert "1" not in df['FAM_target'].unique()
    assert "tagBFP_positive" not in df['sample_description_1'].dropna().unique()

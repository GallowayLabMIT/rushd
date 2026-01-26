import os
from pathlib import Path

import pandas as pd
import pytest

from rushd import qpcr

def test_qpcr_loading(tmp_path: Path):
    """
    Tests that a file with the qPCR default output can be loaded
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )
    with open(str(tmp_path / "data.txt"), "w") as f:
        f.write("""Nonsense first line\nPos\tCp\textra channel\nA1\t1\t2""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = qpcr.load_qpcr_with_metadata(str(tmp_path)+ "/data.txt", yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 1, "A1", "cond1"]]
    df_manual = pd.DataFrame(
        data, columns=["Pos", "Cp", "well", "condition"]
    )
    assert df.equals(df_manual)


def test_qpcr_loading_real_data(tmp_path: Path):
    """
    Tests that a file with the qPCR default output can be loaded,
    copy-pasting from actual qPCR output file
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )
    with open(str(tmp_path / "data.txt"), "w") as f:
        f.write("""Experiment: 2025.08.07_galloway-gaprun-lib-quant_KL  Selected Filter: SYBR Green I / HRM Dye (465-510)
                Include	Color	Pos	Name	Cp	Concentration	Standard	Status
                True	255	A1	Sample 1	27.23		0	""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = qpcr.load_qpcr_with_metadata(str(tmp_path)+ "/data.txt", yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 27.23, "A1", "cond1"]]
    df_manual = pd.DataFrame(
        data, columns=["Pos", "Cp", "well", "condition"]
    )
    assert df.equals(df_manual)
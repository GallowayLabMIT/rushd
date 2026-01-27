import os
from pathlib import Path

import pandas as pd
import pytest

from rushd import qpcr


def test_single_csv(tmp_path: Path):
    """
    Tests that a single file can be read using defaults
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )
    with open(str(tmp_path / "data.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,1,2""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = qpcr.load_single_csv_with_metadata(str(tmp_path) + "/data.csv", yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 1, 2, "cond1"]]
    df_manual = pd.DataFrame(data, columns=["well", "channel1", "channel2", "condition"])
    assert df.equals(df_manual)


def test_single_csv_kwargs(tmp_path: Path):
    """
    Tests that a single file can be read using custom kwargs
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
        f.write("""well\tchannel1\tchannel2\nA1\t1\t2""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = qpcr.load_single_csv_with_metadata(
        str(tmp_path) + "/data.txt", yaml_path, csv_kwargs={"delimiter": "\t"}
    )
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 1, 2, "cond1"]]
    df_manual = pd.DataFrame(data, columns=["well", "channel1", "channel2", "condition"])
    assert df.equals(df_manual)


def test_single_csv_well_column(tmp_path: Path):
    """
    Tests that a single file can be read using custom well column
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )
    with open(str(tmp_path / "data.csv"), "w") as f:
        f.write("""my_well,channel1,channel2\nA1,1,2""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = qpcr.load_single_csv_with_metadata(
        str(tmp_path) + "/data.csv", yaml_path, well_column="my_well"
    )
    df.sort_values(by="my_well", inplace=True, ignore_index=True)

    data = [["A1", 1, 2, "A1", "cond1"]]
    df_manual = pd.DataFrame(data, columns=["my_well", "channel1", "channel2", "well", "condition"])
    assert df.equals(df_manual)
    # Reload specifying columns
    df = qpcr.load_single_csv_with_metadata(
        str(tmp_path) + "/data.csv", yaml_path, well_column="my_well", columns=["channel1"]
    )
    assert "channel1" in df.columns
    assert "channel2" not in df.columns


def test_single_csv_invalid_well_column(tmp_path: Path):
    """
    Tests that a custom well column that is missing from the data correctly raises an error
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )
    with open(str(tmp_path / "data.csv"), "w") as f:
        f.write("""my_well,channel1,channel2\nA1,1,2""")
    yaml_path = str(tmp_path) + "/test.yaml"

    with pytest.raises(qpcr.ColumnError):
        _ = qpcr.load_single_csv_with_metadata(
            str(tmp_path) + "/data.csv", yaml_path, well_column="other_well"
        )


def test_single_csv_invalid_path(tmp_path: Path):
    os.mkdir(tmp_path / "my_dir")
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )

    with pytest.raises(qpcr.DataPathError):
        _ = qpcr.load_single_csv_with_metadata(
            str(tmp_path) + "/my_dir", str(tmp_path) + "/test.yaml"
        )


def test_invalid_yaml_path(tmp_path: Path):
    """
    Tests that invalid .yaml files throw errors
    """
    with pytest.raises(qpcr.YamlError):
        _ = qpcr.load_single_csv_with_metadata("", tmp_path / "nonexistent.yaml")


def test_qpcr_default(tmp_path: Path):
    """
    Tests that a file with the qPCR default output can be loaded,
    and that it overrides previous kwargs
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
    df = qpcr.load_single_csv_with_metadata(
        str(tmp_path) + "/data.txt",
        yaml_path,
        well_column="bad_well",
        columns=["old_col"],
        csv_kwargs={"delimiter": ","},
        is_default=True,
    )
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 1, "A1", "cond1"]]
    df_manual = pd.DataFrame(data, columns=["Pos", "Cp", "well", "condition"])
    assert df.equals(df_manual)


def test_qpcr_default_real_data(tmp_path: Path):
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
        f.write("""Experiment: 2025.08.07_galloway-gaprun-lib-quant_KL  Filter: SYBR Green I
                Include	Color	Pos	Name	Cp	Concentration	Standard	Status
                True	255	A1	Sample 1	27.23		0	""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = qpcr.load_single_csv_with_metadata(str(tmp_path) + "/data.txt", yaml_path, is_default=True)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 27.23, "A1", "cond1"]]
    df_manual = pd.DataFrame(data, columns=["Pos", "Cp", "well", "condition"])
    assert df.equals(df_manual)


def test_plates(tmp_path: Path):
    """
    Tests that several plates can be loaded, with/without filename regex
    """
    # Create data
    sub_dir = ["dir1", "dir2"]
    os.mkdir(tmp_path / sub_dir[0])
    os.mkdir(tmp_path / sub_dir[1])
    with open(str(tmp_path / sub_dir[0] / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / sub_dir[0] / "plate1.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,1,2\nG12,10,20""")

    with open(str(tmp_path / sub_dir[1] / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / sub_dir[1] / "plate2.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,3,4\nG12,30,""" "")

    # Call function
    plates = pd.DataFrame(
        {
            "data_path": [Path(tmp_path / d / f"plate{i+1}.csv") for i, d in enumerate(sub_dir)],
            "yaml_path": [Path(tmp_path / d / "test.yaml") for d in sub_dir],
            "extra_metadata": ["meta1", "meta2"],
        }
    )
    df = qpcr.load_plates_with_metadata(plates)

    # Check against manual output
    data = [
        ["cond1", 1, 2, "A1", "meta1"],
        ["cond1", 10, 20, "G12", "meta1"],
        ["cond1", 3, 4, "A1", "meta2"],
        ["cond1", 30, pd.NA, "G12", "meta2"],
    ]
    df_manual = pd.DataFrame(
        data, columns=["condition", "channel1", "channel2", "well", "extra_metadata"]
    )
    df = df.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(axis="columns")
    df_manual = df_manual.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(
        axis="columns"
    )
    assert df.equals(df_manual)

    df = qpcr.load_plates_with_metadata(plates, filename_regex=r"plate(?P<plate>\d+)\.csv")
    data = [
        ["cond1", 1, 2, "A1", "meta1", "1"],
        ["cond1", 10, 20, "G12", "meta1", "1"],
        ["cond1", 3, 4, "A1", "meta2", "2"],
        ["cond1", 30, pd.NA, "G12", "meta2", "2"],
    ]
    df_manual = pd.DataFrame(
        data, columns=["condition", "channel1", "channel2", "well", "extra_metadata", "plate"]
    )
    df = df.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(axis="columns")
    df_manual = df_manual.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(
        axis="columns"
    )
    assert df.equals(df_manual)


def test_plates_valid_base_path(tmp_path: Path):
    """
    Tests that several plates can be loaded using a valid base_path
    """
    # Create data
    os.mkdir(tmp_path / "dir")
    with open(str(tmp_path / "dir" / "test1.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "dir" / "plate1.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,1,2\nG12,10,20""")

    with open(str(tmp_path / "dir" / "test2.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "dir" / "plate2.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,3,4\nG12,30,40""")

    # Call function
    plates = pd.DataFrame(
        {
            "data_path": ["plate1.csv", "plate2.csv"],
            "yaml_path": ["test1.yaml", "test2.yaml"],
            "extra_metadata": ["meta1", "meta2"],
        }
    )
    df = qpcr.load_plates_with_metadata(plates, base_path=str(tmp_path / "dir"))

    # Check against manual output
    data = [
        ["cond1", 1, 2, "A1", "meta1"],
        ["cond1", 10, 20, "G12", "meta1"],
        ["cond1", 3, 4, "A1", "meta2"],
        ["cond1", 30, 40, "G12", "meta2"],
    ]
    df_manual = pd.DataFrame(
        data, columns=["condition", "channel1", "channel2", "well", "extra_metadata"]
    )
    df = df.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(axis="columns")
    df_manual = df_manual.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(
        axis="columns"
    )
    assert df.equals(df_manual)


def test_plates_df_regex(tmp_path: Path):
    """
    Tests that several plates can be loaded using a valid filename_regex
    for each plate
    """
    # Create data
    os.mkdir(tmp_path / "dir")
    with open(str(tmp_path / "dir" / "test1.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "dir" / "2025_plate1.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,1,2\nG12,10,20""")

    with open(str(tmp_path / "dir" / "test2.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "dir" / "plate2_2026.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,3,4\nG12,30,40""")

    # Call function
    plates = pd.DataFrame(
        {
            "data_path": ["2025_plate1.csv", "plate2_2026.csv"],
            "yaml_path": ["test1.yaml", "test2.yaml"],
            "filename_regex": [
                r"(?P<date>\d+)_plate(?P<plate>\d+)\.csv",
                r"plate(?P<plate>\d+)_(?P<date>\d+)\.csv",
            ],
            "extra_metadata": ["meta1", "meta2"],
        }
    )
    df = qpcr.load_plates_with_metadata(plates, base_path=(tmp_path / "dir"))

    # Check against manual output
    data = [
        ["cond1", 1, 2, "A1", "meta1", "2025", r"(?P<date>\d+)_plate(?P<plate>\d+)\.csv", "1"],
        ["cond1", 10, 20, "G12", "meta1", "2025", r"(?P<date>\d+)_plate(?P<plate>\d+)\.csv", "1"],
        ["cond1", 3, 4, "A1", "meta2", "2026", r"plate(?P<plate>\d+)_(?P<date>\d+)\.csv", "2"],
        ["cond1", 30, 40, "G12", "meta2", "2026", r"plate(?P<plate>\d+)_(?P<date>\d+)\.csv", "2"],
    ]
    df_manual = pd.DataFrame(
        data,
        columns=[
            "condition",
            "channel1",
            "channel2",
            "well",
            "extra_metadata",
            "date",
            "filename_regex",
            "plate",
        ],
    )
    df = df.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(axis="columns")
    df_manual = df_manual.sort_values(by=["extra_metadata", "well"], ignore_index=True).sort_index(
        axis="columns"
    )
    assert df.equals(df_manual)


def test_plates_invalid_regex(tmp_path: Path):
    """
    Tests that error is raised if plate filename doesn't match passed regex
    """
    # Create data
    os.mkdir(tmp_path / "dir")
    with open(str(tmp_path / "dir" / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "dir" / "plate1.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,1,2\nG12,10,20""")
    with open(str(tmp_path / "dir" / "plate2.csv"), "w") as f:
        f.write("""well,channel1,channel2\nA1,3,4\nG12,30,40""")

    # Call function
    plates = pd.DataFrame(
        {
            "data_path": ["plate1.csv", "plate2.csv"],
            "yaml_path": ["test.yaml", "test.yaml"],
            "extra_metadata": ["meta1", "meta2"],
        }
    )
    with pytest.raises(qpcr.RegexError):
        _ = qpcr.load_plates_with_metadata(
            plates,
            base_path=(tmp_path / "dir"),
            filename_regex=r"plate(?P<plate>\d+)_(?P<date>\d+)\.csv",
        )


def test_plates_invalid_df():
    """
    Tests that proper error is thrown when the DataFrame
    specifying plates of data is missing the required columns
    """
    df1 = pd.DataFrame(columns=["yaml_path", "foo"])
    df2 = pd.DataFrame(columns=["bar", "data_path"])
    df3 = pd.DataFrame(columns=["foo"])
    df_list = [df1, df2, df3]
    for df in df_list:
        with pytest.raises(qpcr.GroupsError):
            _ = qpcr.load_plates_with_metadata(df)

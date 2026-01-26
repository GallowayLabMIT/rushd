import os
from pathlib import Path

import pandas as pd
import pytest

from rushd import flow


def test_invalid_yaml_path(tmp_path: Path):
    """
    Tests that invalid .yaml files throw errors
    """
    with pytest.raises(flow.YamlError):
        _ = flow.load_csv_with_metadata("", tmp_path / "nonexistent.yaml")
    with pytest.raises(flow.YamlError):
        _ = flow.load_csv_with_metadata("", "wells")


def test_invalid_yaml_formatting(tmp_path: Path):
    """
    Tests that provided .yaml file improperly formatted without
    all metadata contained under a 'metadata' header throws error
    """
    with open(str(tmp_path / "test0.yaml"), "w") as f:
        f.write(
            """
        -
        """
        )
    with open(str(tmp_path / "test1.yaml"), "w") as f:
        f.write(
            """
        condition:
        - cond1: A1-A4
        """
        )
    with open(str(tmp_path / "test2.yaml"), "w") as f:
        f.write(
            """
        data:
        condition:
        - cond1: A1-A4
        """
        )

    for i in range(3):
        with pytest.raises(flow.YamlError):
            temp_yaml = f"/test{i}.yaml"
            yaml_path = str(tmp_path) + temp_yaml
            _ = flow.load_csv_with_metadata("", yaml_path)


def test_alt_yaml_filenames(tmp_path: Path):
    """
    Tests that alternately-named yaml files (e.g. .yml or something else) loads properly.
    """
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    for yaml_extension in ["yaml", "yml", "something_else"]:
        with open(str(tmp_path / f"test.{yaml_extension}"), "w") as f:
            f.write(
                """
            metadata:
                condition:
                - cond1: A1,G12
            """
            )
        yaml_path = str(tmp_path / f"test.{yaml_extension}")
        df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
        df.sort_values(by="well", inplace=True, ignore_index=True)

        data = [["cond1", "A1", "singlets", 1, 2], ["cond1", "G12", "singlets", 10, 20]]
        df_manual = pd.DataFrame(
            data, columns=["condition", "well", "population", "channel1", "channel2"]
        )
        assert df.equals(df_manual)


def test_arg_types(tmp_path: Path):
    """
    Tests that both str and Path arguments are accepted for yaml and data paths
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")

    yaml_path = str(tmp_path) + "/test.yaml"
    df_str = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df_str.sort_values(by="well", inplace=True, ignore_index=True)

    yaml_path = Path(yaml_path)
    df_path = flow.load_csv_with_metadata(tmp_path, yaml_path)
    df_path.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["cond1", "A1", "singlets", 1, 2], ["cond1", "G12", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=["condition", "well", "population", "channel1", "channel2"]
    )
    assert df_str.equals(df_manual) and df_path.equals(df_manual)


def test_default_regex(tmp_path: Path):
    """
    Tests that files can be read using default file name regular expression
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["cond1", "A1", "singlets", 1, 2], ["cond1", "G12", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=["condition", "well", "population", "channel1", "channel2"]
    )
    assert df.equals(df_manual)


def test_96_well(tmp_path: Path):
    """
    Tests that the maximum extents of a 96-well can be processed.
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,H12
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_H12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["cond1", "A1", "singlets", 1, 2], ["cond1", "H12", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=["condition", "well", "population", "channel1", "channel2"]
    )
    assert df.equals(df_manual)


def test_384_well(tmp_path: Path):
    """
    Tests that the maximum extents of a 384-well can be processed.
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,P24
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_P24_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["cond1", "A1", "singlets", 1, 2], ["cond1", "P24", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=["condition", "well", "population", "channel1", "channel2"]
    )
    assert df.equals(df_manual)


def test_na_for_unspecified_columns(tmp_path: Path):
    """
    Tests that unspecified metadata entries get NA applied
    for the column entry.
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
            second_condition:
            - cond2: A2
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_A2_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["cond1", pd.NA, "A1", "singlets", 1, 2], [pd.NA, "cond2", "A2", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data,
        columns=["condition", "second_condition", "well", "population", "channel1", "channel2"],
    )
    assert df.equals(df_manual)


def test_passed_list_metadata(tmp_path: Path):
    """
    Tests that metadata entries passed as dictionaries (instead of lists)
    give a warning, because these can be sort-dependent and hide duplicate keys.
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
              cond1: A1
              cond2: A2
              cond1: A3
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_A2_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + "/test.yaml"
    with pytest.warns(flow.MetadataWarning):
        _ = flow.load_csv_with_metadata(str(tmp_path), yaml_path)


def test_valid_custom_regex(tmp_path: Path):
    """
    Tests that files can be loaded using valid custom file name
    regular expressions, and that metadata is encoded in the
    output dataframe
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "export_A1_100_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_G12_1000_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")

    regex = r"^.*export_(?P<well>[A-G0-9]+)_(?P<dox>[0-9]+)_(?P<population>.+)\.csv"
    yaml_path = str(tmp_path) + "/test.yaml"
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path, regex)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["cond1", "A1", "100", "singlets", 1, 2], ["cond1", "G12", "1000", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=["condition", "well", "dox", "population", "channel1", "channel2"]
    )
    assert df.equals(df_manual)


def test_invalid_custom_regex(tmp_path: Path):
    """
    Tests that invalid custom file name regular expressions
    throw proper errors
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    regex = r"^.*export_(?P<ID>[A-G0-9]+)_(?P<population>.+)\.csv"
    yaml_path = str(tmp_path / "test.yaml")
    with pytest.raises(flow.RegexError):
        _ = flow.load_csv_with_metadata(str(tmp_path), yaml_path, regex)


def test_no_files(tmp_path: Path):
    """
    Tests that proper error is thrown when no files at the specified path
    fit the file name regular expression
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "bad-name0.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "bad-name1.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + "/test.yaml"
    with pytest.raises(flow.RegexError):
        _ = flow.load_csv_with_metadata(str(tmp_path), yaml_path)


def test_subcolumn_loading(tmp_path: Path):
    """
    Tests that column subsets can be loaded from CSV files.
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + "/test.yaml"
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    # Make sure both are present first
    assert "channel1" in df.columns
    assert "channel2" in df.columns
    # Reload specifying columns
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path, columns=["channel1"])
    assert "channel1" in df.columns
    assert "channel2" not in df.columns


def test_extra_columns(tmp_path: Path):
    """
    Tests that extra unused columns can be specified when loading
    """
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    yaml_path = str(tmp_path) + "/test.yaml"
    # Reload specifying columns
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path, columns=["channel1", "channel3"])
    assert "channel1" in df.columns
    assert "channel2" not in df.columns
    assert "channel3" not in df.columns


def test_group_valid(tmp_path: Path):
    """
    Tests that groups of files can be loaded (no base path)
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
    with open(str(tmp_path / sub_dir[0] / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / sub_dir[0] / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")

    with open(str(tmp_path / sub_dir[1] / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / sub_dir[1] / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n3,4""")
    with open(str(tmp_path / sub_dir[1] / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n30,40""")

    # Call function
    groups = pd.DataFrame(
        {
            "data_path": [Path(tmp_path / d) for d in sub_dir],
            "yaml_path": [Path(tmp_path / d / "test.yaml") for d in sub_dir],
            "extra_metadata": ["meta1", "meta2"],
        }
    )
    df = flow.load_groups_with_metadata(groups)

    # Check against manual output
    data = [
        ["cond1", "A1", "singlets", 1, 2, "meta1"],
        ["cond1", "G12", "singlets", 10, 20, "meta1"],
        ["cond1", "A1", "singlets", 3, 4, "meta2"],
        ["cond1", "G12", "singlets", 30, 40, "meta2"],
    ]
    df_manual = pd.DataFrame(
        data, columns=["condition", "well", "population", "channel1", "channel2", "extra_metadata"]
    )
    df.sort_values(by=["extra_metadata", "well"], inplace=True, ignore_index=True)
    df_manual.sort_values(by=["extra_metadata", "well"], inplace=True, ignore_index=True)
    assert df.equals(df_manual)


def test_group_valid_base_path(tmp_path: Path):
    """
    Tests that groups of files can be loaded with a base path
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
    with open(str(tmp_path / sub_dir[0] / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / sub_dir[0] / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")

    with open(str(tmp_path / sub_dir[1] / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / sub_dir[1] / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n3,4""")
    with open(str(tmp_path / sub_dir[1] / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n30,40""")

    # Call function
    groups = pd.DataFrame(
        {
            "data_path": sub_dir,
            "yaml_path": [str(Path(d) / "test.yaml") for d in sub_dir],
            "extra_metadata": ["meta1", "meta2"],
        }
    )
    df = flow.load_groups_with_metadata(groups, str(tmp_path))

    # Check against manual output
    data = [
        ["cond1", "A1", "singlets", 1, 2, "meta1"],
        ["cond1", "G12", "singlets", 10, 20, "meta1"],
        ["cond1", "A1", "singlets", 3, 4, "meta2"],
        ["cond1", "G12", "singlets", 30, 40, "meta2"],
    ]
    df_manual = pd.DataFrame(
        data, columns=["condition", "well", "population", "channel1", "channel2", "extra_metadata"]
    )
    df.sort_values(by=["extra_metadata", "well"], inplace=True, ignore_index=True)
    df_manual.sort_values(by=["extra_metadata", "well"], inplace=True, ignore_index=True)
    assert df.equals(df_manual)


def test_group_invalid_df():
    """
    Tests that proper error is thrown when the DataFrame
    specifying groups of data is missing the required columns
    """
    df1 = pd.DataFrame(columns=["yaml_path", "foo"])
    df2 = pd.DataFrame(columns=["bar", "data_path"])
    df3 = pd.DataFrame(columns=["foo"])
    df_list = [df1, df2, df3]
    for df in df_list:
        with pytest.raises(flow.GroupsError):
            _ = flow.load_groups_with_metadata(df)


def test_group_custom_regex(tmp_path: Path):
    """
    Tests that groups of files can be loaded with custom regexes specified in
    the input dataframe.
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
    with open(str(tmp_path / sub_dir[0] / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / sub_dir[0] / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")

    with open(str(tmp_path / sub_dir[1] / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: B1,H12
        """
        )
    with open(str(tmp_path / sub_dir[1] / "export_plate1_293T_B1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n3,4""")
    with open(str(tmp_path / sub_dir[1] / "export_plate1_293T_H12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n30,40""")

    # Call function
    default_regex = r"^.*export_(?P<well>[A-P]\d+)_(?P<population>.+)\.csv"
    custom_regex = (
        r"^.*export_plate(?P<plate>\d)_(?P<cell>\w+)_(?P<well>[A-P]\d+)_(?P<population>.+)\.csv"
    )
    groups = pd.DataFrame(
        {
            "data_path": sub_dir,
            "yaml_path": [str(Path(d) / "test.yaml") for d in sub_dir],
            "filename_regex": [default_regex, custom_regex],
            "extra_metadata": ["meta1", "meta2"],
        }
    )
    df = flow.load_groups_with_metadata(groups, str(tmp_path))

    # Check against manual output
    data = [
        ["cond1", "A1", "singlets", pd.NA, pd.NA, 1, 2, "meta1", default_regex],
        ["cond1", "G12", "singlets", pd.NA, pd.NA, 10, 20, "meta1", default_regex],
        ["cond1", "B1", "singlets", "1", "293T", 3, 4, "meta2", custom_regex],
        ["cond1", "H12", "singlets", "1", "293T", 30, 40, "meta2", custom_regex],
    ]
    df_manual = pd.DataFrame(
        data,
        columns=[
            "condition",
            "well",
            "population",
            "plate",
            "cell",
            "channel1",
            "channel2",
            "extra_metadata",
            "filename_regex",
        ],
    )
    df.sort_values(by=["extra_metadata", "well"], inplace=True, ignore_index=True)
    df.sort_index(axis=1, inplace=True)
    df_manual.sort_values(by=["extra_metadata", "well"], inplace=True, ignore_index=True)
    df_manual.sort_index(axis=1, inplace=True)
    pd.testing.assert_frame_equal(df_manual, df)


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
    df = flow.load_single_csv_with_metadata(str(tmp_path)+ "/data.csv", yaml_path)
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 1, 2, "cond1"]]
    df_manual = pd.DataFrame(
        data, columns=["well", "channel1", "channel2", "condition"]
    )
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
    df = flow.load_single_csv_with_metadata(str(tmp_path)+ "/data.txt", yaml_path, csv_kwargs=dict(sep='\t'))
    df.sort_values(by="well", inplace=True, ignore_index=True)

    data = [["A1", 1, 2, "cond1"]]
    df_manual = pd.DataFrame(
        data, columns=["well", "channel1", "channel2", "condition"]
    )
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
    df = flow.load_single_csv_with_metadata(str(tmp_path)+ "/data.csv", yaml_path, well_column='my_well')

    data = [["A1", 1, 2, "A1", "cond1"]]
    df_manual = pd.DataFrame(
        data, columns=["my_well", "channel1", "channel2", "well", "condition"]
    )
    assert df.equals(df_manual)
    # Reload specifying columns
    df = flow.load_single_csv_with_metadata(str(tmp_path)+ "/data.csv", yaml_path, well_column='my_well',
                                            columns=['channel1'])
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
    
    with pytest.raises(flow.ColumnError):
        _ = flow.load_single_csv_with_metadata(str(tmp_path) + "/data.csv", yaml_path, well_column='other_well')


def test_single_csv_invalid_path(tmp_path: Path):
    os.mkdir(tmp_path/'my_dir')
    with open(str(tmp_path / "test.yaml"), "w") as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
        """
        )

    with pytest.raises(flow.DataPathError):
        _ = flow.load_single_csv_with_metadata(str(tmp_path) + "/my_dir", str(tmp_path) + "/test.yaml")


def test_csv_no_metadata(tmp_path: Path):
    """
    Tests that files can be loaded without additional metadata
    """
    with open(str(tmp_path / "export_A1_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_G12_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")
    df = flow.load_csv(str(tmp_path))

    data = [["A1", "singlets", 1, 2], ["G12", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=["condition", "population", "channel1", "channel2"]
    )
    assert df.equals(df_manual)


def test_csv_valid_custom_regex(tmp_path: Path):
    """
    Tests that files can be loaded using valid custom file name
    regular expressions, and that the metadata is included in the
    output dataframe
    """
    with open(str(tmp_path / "export_BFP_100_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / "export_GFP_1000_singlets.csv"), "w") as f:
        f.write("""channel1,channel2\n10,20""")

    regex = r"^.*export_(?P<construct>.+)_(?P<dox>[0-9]+)_(?P<population>.+)\.csv"
    df = flow.load_csv(str(tmp_path), regex)
    df.sort_values(by="construct", inplace=True, ignore_index=True)

    data = [["BFP", "100", "singlets", 1, 2], ["GFP", "1000", "singlets", 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=["construct", "dox", "population", "channel1", "channel2"]
    )
    assert df.equals(df_manual)


from pathlib import Path

import pandas as pd
import pytest

from rushd import flow


def test_invalid_yaml_path(tmp_path: Path):
    """
    Tests that invalid .yaml files throw errors
    """
    with pytest.raises(flow.YamlError):
        _ = flow.load_csv_with_metadata('', tmp_path / 'nonexistent.yaml')
    with pytest.raises(flow.YamlError):
        _ = flow.load_csv_with_metadata('', 'wells')


def test_invalid_yaml_formatting(tmp_path: Path):
    """
    Tests that provided .yaml file improperly formatted without
    all metadata contained under a 'metadata' header throws error
    """
    with open(str(tmp_path / 'test0.yaml'), 'w') as f:
        f.write(
            """
        -
        """
        )
    with open(str(tmp_path / 'test1.yaml'), 'w') as f:
        f.write(
            """
        condition:
        - cond1: A1-A4
        """
        )
    with open(str(tmp_path / 'test2.yaml'), 'w') as f:
        f.write(
            """
        data:
        condition:
        - cond1: A1-A4
        """
        )

    for i in range(3):
        with pytest.raises(flow.YamlError):
            temp_yaml = f'/test{i}.yaml'
            yaml_path = str(tmp_path) + temp_yaml
            _ = flow.load_csv_with_metadata('', yaml_path)


def test_alt_yaml_filenames(tmp_path: Path):
    """
    Tests that alternately-named yaml files (e.g. .yml or something else) loads properly.
    """
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_G12_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    for yaml_extension in ['yaml', 'yml', 'something_else']:
        with open(str(tmp_path / f'test.{yaml_extension}'), 'w') as f:
            f.write(
                """
            metadata:
                condition:
                - cond1: A1,G12
            """
            )
        yaml_path = str(tmp_path / f'test.{yaml_extension}')
        df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
        df.sort_values(by='well', inplace=True, ignore_index=True)

        data = [['cond1', 'A1', 'singlets', 1, 2], ['cond1', 'G12', 'singlets', 10, 20]]
        df_manual = pd.DataFrame(
            data, columns=['condition', 'well', 'population', 'channel1', 'channel2']
        )
        assert df.equals(df_manual)


def test_arg_types(tmp_path: Path):
    """
    Tests that both str and Path arguments are accepted for yaml and data paths
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_G12_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")

    yaml_path = str(tmp_path) + '/test.yaml'
    df_str = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df_str.sort_values(by='well', inplace=True, ignore_index=True)

    yaml_path = Path(yaml_path)
    df_path = flow.load_csv_with_metadata(tmp_path, yaml_path)
    df_path.sort_values(by='well', inplace=True, ignore_index=True)

    data = [['cond1', 'A1', 'singlets', 1, 2], ['cond1', 'G12', 'singlets', 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=['condition', 'well', 'population', 'channel1', 'channel2']
    )
    assert df_str.equals(df_manual) and df_path.equals(df_manual)


def test_default_regex(tmp_path: Path):
    """
    Tests that files can be read using default file name regular expression
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_G12_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + '/test.yaml'
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by='well', inplace=True, ignore_index=True)

    data = [['cond1', 'A1', 'singlets', 1, 2], ['cond1', 'G12', 'singlets', 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=['condition', 'well', 'population', 'channel1', 'channel2']
    )
    assert df.equals(df_manual)


def test_96_well(tmp_path: Path):
    """
    Tests that the maximum extents of a 96-well can be processed.
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,H12
        """
        )
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_H12_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + '/test.yaml'
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by='well', inplace=True, ignore_index=True)

    data = [['cond1', 'A1', 'singlets', 1, 2], ['cond1', 'H12', 'singlets', 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=['condition', 'well', 'population', 'channel1', 'channel2']
    )
    assert df.equals(df_manual)


def test_384_well(tmp_path: Path):
    """
    Tests that the maximum extents of a 384-well can be processed.
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,P24
        """
        )
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_P24_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + '/test.yaml'
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by='well', inplace=True, ignore_index=True)

    data = [['cond1', 'A1', 'singlets', 1, 2], ['cond1', 'P24', 'singlets', 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=['condition', 'well', 'population', 'channel1', 'channel2']
    )
    assert df.equals(df_manual)


def test_na_for_unspecified_columns(tmp_path: Path):
    """
    Tests that unspecified metadata entries get NA applied
    for the column entry.
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1
            second_condition:
            - cond2: A2
        """
        )
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_A2_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + '/test.yaml'
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path)
    df.sort_values(by='well', inplace=True, ignore_index=True)

    data = [['cond1', pd.NA, 'A1', 'singlets', 1, 2], [pd.NA, 'cond2', 'A2', 'singlets', 10, 20]]
    df_manual = pd.DataFrame(
        data,
        columns=['condition', 'second_condition', 'well', 'population', 'channel1', 'channel2'],
    )
    assert df.equals(df_manual)


def test_passed_list_metadata(tmp_path: Path):
    """
    Tests that metadata entries passed as dictionaries (instead of lists)
    give a warning, because these can be sort-dependent and hide duplicate keys.
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
              cond1: A1
              cond2: A2
              cond1: A3
        """
        )
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_A2_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + '/test.yaml'
    with pytest.warns(flow.MetadataWarning):
        _ = flow.load_csv_with_metadata(str(tmp_path), yaml_path)


def test_valid_custom_regex(tmp_path: Path):
    """
    Tests that files can be loaded using valid custom file name
    regular expressions, and that metadata is encoded in the
    output dataframe
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / 'export_A1_100_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_G12_1000_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")

    regex = r'^.*export_(?P<well>[A-G0-9]+)_(?P<dox>[0-9]+)_(?P<population>.+)\.csv'
    yaml_path = str(tmp_path) + '/test.yaml'
    df = flow.load_csv_with_metadata(str(tmp_path), yaml_path, regex)
    df.sort_values(by='well', inplace=True, ignore_index=True)

    data = [['cond1', 'A1', '100', 'singlets', 1, 2], ['cond1', 'G12', '1000', 'singlets', 10, 20]]
    df_manual = pd.DataFrame(
        data, columns=['condition', 'well', 'dox', 'population', 'channel1', 'channel2']
    )
    print(df)
    print(df_manual)
    assert df.equals(df_manual)


def test_invalid_custom_regex(tmp_path: Path):
    """
    Tests that invalid custom file name regular expressions
    throw proper errors
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / 'export_A1_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'export_G12_singlets.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    regex = r'^.*export_(?P<ID>[A-G0-9]+)_(?P<population>.+)\.csv'
    yaml_path = str(tmp_path / 'test.yaml')
    with pytest.raises(flow.RegexError):
        _ = flow.load_csv_with_metadata(str(tmp_path), yaml_path, regex)


def test_no_files(tmp_path: Path):
    """
    Tests that proper error is thrown when no files at the specified path
    fit the file name regular expression
    """
    with open(str(tmp_path / 'test.yaml'), 'w') as f:
        f.write(
            """
        metadata:
            condition:
            - cond1: A1,G12
        """
        )
    with open(str(tmp_path / 'bad-name0.csv'), 'w') as f:
        f.write("""channel1,channel2\n1,2""")
    with open(str(tmp_path / 'bad-name1.csv'), 'w') as f:
        f.write("""channel1,channel2\n10,20""")
    yaml_path = str(tmp_path) + '/test.yaml'
    with pytest.raises(flow.RegexError):
        _ = flow.load_csv_with_metadata(str(tmp_path), yaml_path)

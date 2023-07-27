from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rushd.plot
from pytest_mock import MockerFixture

TEST_METADATA = """
metadata:
    inducible_fluorophore:
        - TagBFP: A1-H9
        - EGFP: A10-H12
    inducible_spacerlength:
        - 0: A1-H9
        - 2: A10-H12
    test_negative_linear:
        - -1: A1-H9
        - 2: A10-H12
    orientation:
        - tandem: A1-H3, A10-H12
        - convergent: A4-H6
        - divergent: A7-H9
    test_empty:
        - wells: A1-C4
    dox:
        - 0: A1-A12
        - 0.0000820: B1-B12
        - 0.0003248: C1-C12
        - 0.0012864: D1-D12
        - 0.0050940: E1-E12
        - 0.0201723: F1-F12
        - 0.0798822: G1-G12
        - 0.3163330: H1-H12
"""


def test_plot_n_outputs(tmp_path: Path, mocker: MockerFixture):
    """
    Tests that the plot_well_metadata function properly outputs multiple plots, to both the screen
    and to a given output directory.
    """
    with (tmp_path / "metadata.yaml").open("w") as meta_file:
        meta_file.write(TEST_METADATA)

    show_mock = mocker.patch("matplotlib.pyplot.show")
    rushd.plot.plot_well_metadata(tmp_path / "metadata.yaml", columns=["dox"])
    show_mock.assert_called_once()
    show_mock.reset_mock()
    rushd.plot.plot_well_metadata(tmp_path / "metadata.yaml")
    show_mock.assert_called()
    out_path = tmp_path / "output"
    out_path.mkdir()
    rushd.plot.plot_well_metadata(tmp_path / "metadata.yaml", output_dir=out_path)
    assert len(list(out_path.glob("*"))) == 6 * 3


def test_column_autodetection(tmp_path: Path):
    """
    Tests the autodetection of numerical and categorical columns, by comparing legend colors
    """
    with (tmp_path / "metadata.yaml").open("w") as meta_file:
        meta_file.write(TEST_METADATA)
    mapping = rushd.flow.load_well_metadata(tmp_path / "metadata.yaml")

    rushd.plot.plot_mapping(mapping["dox"])
    autodetected_log = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    rushd.plot.plot_mapping(mapping["dox"], style="log")
    manual_log = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    assert autodetected_log == manual_log

    rushd.plot.plot_mapping(mapping["inducible_spacerlength"])
    autodetected_linear = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    rushd.plot.plot_mapping(mapping["inducible_spacerlength"], style="linear")
    manual_linear = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    assert autodetected_linear == manual_linear

    rushd.plot.plot_mapping(mapping["test_negative_linear"])
    autodetected_linear = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    rushd.plot.plot_mapping(mapping["test_negative_linear"], style="linear")
    manual_linear = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    assert autodetected_linear == manual_linear

    rushd.plot.plot_mapping(mapping["orientation"])
    autodetected_category = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    rushd.plot.plot_mapping(mapping["orientation"], style="category")
    manual_category = [p.get_facecolor() for p in plt.gca().get_legend().get_patches()]
    plt.close()
    assert autodetected_category == manual_category


def test_custom_style(tmp_path: Path):
    """
    Test that custom styles can be passed to be plotted.
    """
    with (tmp_path / "metadata.yaml").open("w") as meta_file:
        meta_file.write(TEST_METADATA)
    mapping = rushd.flow.load_well_metadata(tmp_path / "metadata.yaml")

    custom_style = {
        "tandem": (1.0, 0.0, 0.0, 1.0),
        "convergent": (0.0, 1.0, 0.0, 1.0),
        "divergent": (0.0, 0.0, 1.0, 1.0),
    }
    rushd.plot.plot_mapping(mapping["orientation"], style=custom_style)
    color_text_mapping = {
        t.get_text(): p.get_facecolor()
        for t, p in zip(plt.gca().get_legend().get_texts(), plt.gca().get_legend().get_patches())
    }
    plt.close()
    assert custom_style == color_text_mapping


def test_larger_plates(tmp_path: Path):
    """
    Test that larger plates can be processed
    """
    with (tmp_path / "metadata.yaml").open("w") as meta_file:
        meta_file.write("metadata:\n  example:\n    - test: A1-G14")
    mapping = rushd.flow.load_well_metadata(tmp_path / "metadata.yaml")
    rushd.plot.plot_mapping(mapping["example"])
    plt.close()


def test_single_entry_zero(tmp_path: Path):
    """Tests that single entries where max=min do not crash the plotter"""
    with (tmp_path / "metadata.yaml").open("w") as meta_file:
        meta_file.write("metadata:\n  foo:\n    - 0: A1-H6")
    mapping = rushd.flow.load_well_metadata(tmp_path / "metadata.yaml")
    rushd.plot.plot_mapping(mapping["foo"])
    plt.close()


def test_generate_xticklabels():
    """Tests that xticklabels are replaced as expected"""
    df_labels = pd.DataFrame(
        {"category": ["cat_A", "cat_B"], "metadata1": ["foo", "bar"], "metadata2": ["-", "+"]}
    )
    plt.plot(["cat_A", "cat_B"], [0, 1])
    rushd.plot.generate_xticklabels(df_labels, "category", ["metadata1", "metadata2"])
    new_labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    expected_labels = ["foo\n-", "bar\n+"]
    plt.close()
    assert new_labels == expected_labels

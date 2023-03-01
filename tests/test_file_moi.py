from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from rushd.flow import MOIinputError, moi


def generate_art_data(seed, moi, replicate=1) -> pd.DataFrame:
    """
    Generates an artificial data set of 50000 cells, 10000 at each titration level,
    assuming the input moi.
    """
    rstate = np.random.RandomState(seed)
    art_data: List[pd.DataFrame] = []
    for scale in [1, 0.1, 0.01, 0.001, 0.0001]:
        cells = rstate.poisson(scale * moi, size=10000)
        temp_df = pd.DataFrame(
            {
                "condition": np.ones(cells.shape),
                "replicate": replicate * np.ones(cells.shape),
                "starting_cell_count": 10000 * np.ones(cells.shape),
                "scaling": scale * np.ones(cells.shape),
                "max_virus": np.ones(cells.shape),
                "color": 1000 * cells,
            }
        )
        art_data.append(temp_df)
    return pd.concat(art_data, ignore_index=True)


def test_invalid_dataframe():
    """
    Tests that a dataframe without the proper columns throw errors
    """
    local_df = generate_art_data(5, 1)
    no_cond = local_df.drop(columns=["condition"])
    with pytest.raises(MOIinputError):
        _ = moi(no_cond, "color", 0)
    no_color = local_df.drop(columns=["color"])
    with pytest.raises(MOIinputError):
        _ = moi(no_color, "color", 0)
    multi_gone = local_df.drop(columns=["replicate", "starting_cell_count"])
    with pytest.raises(MOIinputError):
        _ = moi(multi_gone, "color", 0)


def test_moi_math(tmp_path: Path):
    """
    Tests that the proper math is being done.
    """
    local_df = generate_art_data(5, 1)
    outcome = moi(local_df, "color", 0, output_path=tmp_path)
    assert abs(outcome["moi"].iloc[0] - 1) <= 0.05
    local_df = generate_art_data(5, 2.3)
    outcome = moi(local_df, "color", 0, output_path=tmp_path)
    assert abs(outcome["moi"].iloc[0] - 2.3) <= 0.05
    local_df = generate_art_data(5, 0.1)
    outcome = moi(local_df, "color", 0, output_path=tmp_path)
    assert abs(outcome["moi"].iloc[0] - 0.1) <= 0.05


def test_makes_MOIgraphs(tmp_path: Path):
    """
    Tests that the MOI graphs are created
    """
    local_df = generate_art_data(5, 1)
    moi(local_df, "color", 0, output_path=tmp_path)
    file_path = tmp_path / "figures" / "1.0_MOIcurve.png"
    assert file_path.is_file()


def test_makes_titergraphs(tmp_path: Path):
    """
    Tests that the titer graphs are created
    """
    local_df = generate_art_data(5, 1)
    moi(local_df, "color", 0, output_path=tmp_path)
    file_path = tmp_path / "figures" / "1.0_titer.png"
    assert file_path.is_file()


def test_mean_median(tmp_path: Path):
    """
    Tests that switching between mean and median work properly
    """
    temp_df1 = generate_art_data(5, 1, 1)
    temp_df2 = generate_art_data(5, 1, 2)
    temp_df3 = generate_art_data(5, 2, 3)
    local_df = pd.concat([temp_df1, temp_df2, temp_df3], ignore_index=True)
    outcome1 = moi(local_df, "color", 0, output_path=tmp_path, summary_method="mean")
    assert abs(outcome1["moi"].iloc[0] - 1.33) <= 0.05
    outcome2 = moi(local_df, "color", 0, output_path=tmp_path, summary_method="median")
    assert abs(outcome2["moi"].iloc[0] - 1) <= 0.05


def test_shows_plot(mocker: MockerFixture):
    """
    Tests that no output path results in displayed plots
    """
    local_df = generate_art_data(5, 1)
    show_mock = mocker.patch("matplotlib.pyplot.show")
    moi(local_df, "color", 0)
    show_mock.assert_called()

from pathlib import Path

import pandas as pd
import rushd.io


def test_pandas_cache_decorator(tmp_path: Path):
    """
    Tests that the caching decorator works
    """
    n_regens: int = 0

    @rushd.io.cache_dataframe(tmp_path / "dfcache.gzip")
    def gen_dataframe() -> pd.DataFrame:
        nonlocal n_regens
        n_regens += 1
        return pd.DataFrame({"test": [1, 2, 3]})

    assert n_regens == 0
    _ = gen_dataframe()
    assert n_regens == 1
    _ = gen_dataframe()
    assert n_regens == 1
    _ = gen_dataframe(invalidate=True)
    assert n_regens == 2

    @rushd.io.cache_dataframe(str(tmp_path / "dfcache2.gzip"))
    def gen_dataframe_str() -> pd.DataFrame:
        nonlocal n_regens
        n_regens += 1
        return pd.DataFrame({"test": [1, 2, 3]})

    _ = gen_dataframe_str()
    assert n_regens == 3
    _ = gen_dataframe_str()
    assert n_regens == 3

import pytest
from rushd.well_mapper import well_mapping


def test_default_separator():
    """
    Tests that the default separator is a period,
    and that conditions are properly merged together.
    """
    result = well_mapping([{"foo": "A1"}, {"bar": "A1"}])
    print(result)
    assert result["A01"] == "foo.bar"


def test_custom_separator():
    """
    Tests that we can override the mapping separator.
    """
    for sep in r"!@#$%^&*()<>,\/":
        result = well_mapping([{"foo": "A1"}, {"bar": "A1"}], separator=sep)
        assert result["A01"] == f"foo{sep}bar"


def test_combo_key_type():
    """
    Tests that we can combine strings and nonstrings with a separator.
    """
    result = well_mapping([{0: "A1"}, {"bar": "A1"}])
    print(result)
    assert result["A01"] == "0.bar"


def test_valid_mapping_spec():
    """
    Tests valid specifications do not throw an error
    """
    _ = well_mapping(
        {
            "a": "A01",
            "b": "A1",
            "c": "A2,",  # allow trailing commas
            "d": "A1-B12",
            "e": "A1-B12,C5,C4-F8",
            "f": "A1-B12, C12, D4",  # allow whitespace
        }
    )
    assert True


def test_invalid_mapping_spec():
    """
    Tests that invalid specifications throw errors
    """
    with pytest.raises(ValueError):
        _ = well_mapping({"a": ""})
    with pytest.raises(ValueError):
        _ = well_mapping({"a": "Z99"})
    with pytest.raises(ValueError):
        _ = well_mapping({"a": "A1:A15"})


def test_backwards_rectangles():
    """
    Tests that arbitrary rectangles
    are allowed (even those that are not
    upper-left corner to bottom-right)
    """
    result = well_mapping([{"foo": "F8-C4"}])
    for key in ["C4", "C8", "F4", "F8", "D6"]:
        assert result[key] == "foo"


def test_normed_and_unnormed_single_well():
    """
    Tests that normalized and un-normalized well-IDs
    are handled for looking up a single well entry.
    """
    result = well_mapping([{"foo": "A1"}, {"bar": "A10"}, {"baz": "A1,A10"}])
    assert result["A1"] == "foo.baz"
    assert result["A01"] == "foo.baz"
    assert result["A10"] == "bar.baz"


def test_normed_and_unnormed_rectangle():
    """
    Tests that normalized and un-normalized well-IDs
    are handled for looking up a rectangular mapping entry.
    """
    result = well_mapping([{"foo": "A1-A5"}, {"bar": "A6-A10"}, {"baz": "A1-A10"}])
    assert result["A1"] == "foo.baz"
    assert result["A01"] == "foo.baz"
    assert result["A10"] == "bar.baz"


def test_normed_and_unnormed_mix():
    """
    Tests that normalized and un-normalized well-IDs
    are handled for looking up a mix of mapping entries.
    """
    result = well_mapping([{"foo": "A1-A5"}, {"bar": "A6-A10"}, {"baz": "A1,A10"}])
    assert result["A1"] == "foo.baz"
    assert result["A01"] == "foo.baz"
    assert result["A10"] == "bar.baz"


def test_normed_unnormed_input():
    """
    Tests that normalized and unnormalized input well mappings work.
    """
    result = well_mapping(
        [{"foo": "A1-G9"}, {"bar": "A01-G09"}, {"baz": "A1-G09"}, {"qaz": "A01-G9"}]
    )
    for i in range(1, 10):
        assert result[f"A{i}"] == "foo.bar.baz.qaz"
        assert result[f"A{i:02d}"] == "foo.bar.baz.qaz"

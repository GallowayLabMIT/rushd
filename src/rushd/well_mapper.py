"""Converts user-specified plate specifications into well maps.

Rationale
---------
Helper module that parses plate specifications of the form:
```yaml
MEF-low: A1-E1
MEF-bulk: F1-H1, A2-H2, A3-B3
retroviral: A1-H12
```
and returns a dictionary that lets you map from well number to a
plate specification.

This format allows for robust and concise description of plate maps.

Specification
-------------
While these plate maps can be concisely defined inside YAML or JSON
files, this specification does not define an underlying format; it only
deals with how to handle the specification.

A *well specification* is a string containing a comma-separated list of
*region specifiers*. A region specifier is one of two forms, a single
well form:

```
    A1
    B05
```

or a rectangular region form:

```
    A1-A12
    B05-D8
    B05 - C02
```

As seen in these examples, the rectangular region form is distinguished
by the presence of a hyphen between two single-well identifiers. Whitespace
and leading zeros are allowed.

A well specification is first *normalized* by the software, where all whitespace
characters are removed. The resulting string is split by commas, and further parsed
as one of the region specifiers.

Within a single specifier, duplicate entries are *ignored*. That is, the following
specifiers are all equivalent:

```
    A5-B7
    A5,A6,A7,B5,B6,B7
    A5-B7,B6
    A5-B7,B5-B7
```

A *plate specification* is either a dictionary (if order is not important)
or a sequence of dictionaries (if order is important). The difference between these
in a YAML underlying format is:

```yaml
test: A5-A7
test2: A5-A9
```

which yields `{'test': 'A5-A7', 'test2': 'A5-A9'}`
and

```yaml
- test: A5-A7
- test2: A5-A9
```

which yields `[{'test': 'A5-A7'}, {'test2': 'A5-A9'}]`

This module reads either of these formats. It iterates over each of the well specifications,
building up a dictionary that maps wells to conditions. If multiple well specifications overlap,
then condition names are merged in the order in which they appear, separated by a separator
(by default, a period). This allows very concise condition layouts, such as the following:

```yaml
conditions:
    MEF: A1-C12
    293: D1-F12
    untransformed: A1-D3
    experimental: A4-D12
```

will return a well map of the form:

```
{'A1': 'MEF.untransformed', ..., 'C10: 293.experimental'}
```

Both the non-normalized (e.g. no leading zeros, `A1`) and normalized
(e.g. with leading zeros, `A01`) forms are returned for mapping.
"""
import itertools
import re
from typing import Any, Dict, List, Set, Tuple, Union


def well_mapping(
    plate_spec: Union[Dict[Any, str], List[Dict[Any, str]], Tuple[Dict[Any, str]]],
    separator: str = ".",
) -> Dict[str, Any]:
    """Generate a well mapping given a plate specification.

    Parameters
    ----------
    plate_spec: dict or iterable
        Either a single dictionary containing well specifications,
        or an iterable (list, tuple, etc) that returns dictionaries
        or well specifications as items.
    separator: str
        The separator to use for overlapping plate specifications

    Returns
    -------
    A dictionary that maps wells to conditions.
    """
    # Save plate_spec into a list of a single dictionary if it isn't already
    # an iterable of dictionaries
    if isinstance(plate_spec, Dict):
        plate_spec = [plate_spec]

    # Char To Int mapping and Int To Char mapping
    cti_mapping = {v: k for k, v in enumerate(list("ABCDEFGHIJKLMNOP"))}
    itc_mapping = dict(
        enumerate(list("ABCDEFGHIJKLMNOP"))
    )  # pylint: disable=unnecessary-comprehension

    output_mapping: Dict[str, Any] = {}
    for mapping_dict in plate_spec:
        for key, val in mapping_dict.items():
            if len(val) == 0:
                raise ValueError("Empty mapping spec is not allowed!")
            # Remove all whitespace
            # Allow trailing commas
            tokenized = "".join(val.split()).rstrip(",").split(",")

            wells: Set[str] = set()
            for token in tokenized:
                single_result = re.fullmatch(r"^([A-P]\d+)$", token)
                dual_result = re.fullmatch(r"^([A-P]\d+)-([A-P]\d+)$", token)
                if single_result is None and dual_result is None:
                    raise ValueError(f"Invalid mapping spec: {key}:{val}, problem spec: {token}")

                if single_result is not None:
                    # Add a single well to the well mapping. Add both the normalized and
                    # non-normalized versions
                    wells.add(f"{single_result.group(1)[0]}{int(single_result.group(1)[1:]):02d}")
                    wells.add(f"{single_result.group(1)[0]}{int(single_result.group(1)[1:])}")
                elif dual_result is not None:
                    # Iterate over all wells
                    corners = [
                        (
                            cti_mapping[dual_result.group(i)[0]],
                            int(dual_result.group(i)[1:]),
                        )
                        for i in range(1, 3)
                    ]
                    for well in itertools.product(
                        range(
                            min(corners[0][0], corners[1][0]),
                            max(corners[0][0], corners[1][0]) + 1,
                        ),
                        range(
                            min(corners[0][1], corners[1][1]),
                            max(corners[0][1], corners[1][1]) + 1,
                        ),
                    ):
                        # Add the non-normalized and normalized (leading zeros)
                        # versions. Adding an existing entry is a no-op, so no
                        # harm done.
                        wells.add(f"{itc_mapping[well[0]]}{well[1]}")
                        wells.add(f"{itc_mapping[well[0]]}{well[1]:02d}")
            for parsed_well in wells:
                if parsed_well not in output_mapping:
                    output_mapping[parsed_well] = key
                else:
                    output_mapping[parsed_well] = (
                        str(output_mapping[parsed_well]) + separator + str(key)
                    )
    return output_mapping

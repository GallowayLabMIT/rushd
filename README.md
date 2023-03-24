# rushd
[![PyPI-downloads](https://img.shields.io/pypi/dm/rushd)](https://pypi.org/project/rushd)
[![PyPI-version](https://img.shields.io/pypi/v/rushd)](https://pypi.org/project/rushd)
[![PyPI-license](https://img.shields.io/pypi/l/rushd)](https://pypi.org/project/rushd)
[![Supported python versions](https://img.shields.io/pypi/pyversions/rushd)](https://pypi.org/project/rushd)
[![codecov](https://codecov.io/gh/GallowayLabMIT/rushd/branch/main/graph/badge.svg?token=ALaU8lQxt5)](https://codecov.io/gh/GallowayLabMIT/rushd)

A package for maintaining robust, reproducible data management.

## Rationale
Science relies on repeatable results. `rushd` is a Python package that helps with this, both by making sure that the execution context (e.g. the state of all of the Pip packages) is saved, alongside helper functions that help you cleanly, but repeatedly, separate data from code.

## Install
This package is on Pip, so you can just:
```
pip install rushd
```

Alternatively, you can get built wheels from the [Releases tab on Github](https://github.com/GallowayLabMIT/rushd/releases).

## Quickstart
Simply import `rushd`!
```
import rushd as rd
```

## Complete Examples

## Developer install
If you'd like to hack locally on `rushd`, after cloning this repository:
```
$ git clone https://github.com/GallowayLabMIT/rushd.git
$ cd rushd
```
you can create a local virtual environment, and install `rushd` in "development (editable) mode"
with the extra requirements for tests.
```
$ python -m venv env
$ .\env\Scripts\activate    (on Windows)
$ source env/bin/activate   (on Mac/Linux)
$ pip install -e .[dev]
```
After this 'local install', you can use and import `rushd` freely without
having to re-install after each update.

## Changelog
See the [CHANGELOG](CHANGELOG.md) for detailed changes.
```
## [0.3.0] - 2023-03-01
### Added
- `rd.flow.load_groups_with_metadata` loads files from several folders (groups, e.g. corresponding to plates) into a single dataframe
### Modified
- `rd.flow.moi` properly generates plots of MOI vs fraction infected for checking calculation accuracy
- `rd.well_mapper` properly handles metadata concatenation
```

## License
This is licensed by the MIT license. Use freely!

## What does the name mean?
The name is a reference to [Ibn Rushd](https://en.wikipedia.org/wiki/Averroes), a Muslim scholar born in Córdoba who was responsible for translating and adding scholastic commentary to ancient Greek works, especially Aristotle. His translations spurred further translations into Latin and Hebrew, reigniting interest in ancient Greek works for the first time since the fall of the Roman empire.

His name is pronounced [rush-id](https://translate.google.com/?sl=auto&tl=en&text=%20%D8%A7%D8%A8%D9%86%20%D8%B1%D8%B4%D8%AF&op=translate).

If we take the first and last letter, we also get `rd`: repeatable data!

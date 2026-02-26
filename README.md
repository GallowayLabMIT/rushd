# rushd
[![Stable documentation](https://img.shields.io/badge/Documentation-stable-blue)](https://gallowaylabmit.github.io/rushd/en/main/)
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

## Documentation
See the documentation available at https://gallowaylabmit.github.io/rushd

## Developer install and contributing
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
$ pip install -e .[dev]     (on most shells)
$ pip install -e '.[dev]'   (on zsh)
```
After this 'local install', you can use and import `rushd` freely without
having to re-install after each update.

### Pre-commit
We use something called [pre-commit](https://pre-commit.com/) to automatically
run linters, formatters, and other checks to make sure the code stays high quality.

After doing the developer install and activating the virtual environment, you should run:
```
$ pre-commit install
```
to install the git hooks. Now, pre-commit will automatically run whenever you go to commit.

### Testing with pytest
We use [pytest](https://docs.pytest.org/en/stable/) to test our code. You just type:
```
$ pytest
```
to run all tests, though you can add an optional argument to run some subset of the tests:
```
$ pytest tests/test_file_io.py
```

Pytest automatically discovers tests put in the `tests` directory, whose files and functions
start with the word `test`.

### Code coverage
On every push, all of the tests are run and the **coverage**, or which lines are "covered"
or executed during all tests, is calculated and uploaded to
[Codecov](https://app.codecov.io/github/GallowayLabMIT/rushd). This is a nice way of
seeing if you missed any edge cases that need tests added.


### Publishing a release
Following the steps described above, the full process for publishing a release is:

1. Test

    - Write new tests as needed
    - Run tests  to confirm changes pass

2. Pre-commit

    - Stage changes in git
    - Run `pre-commit` (requires developer mode)
    - Resolve any errors/warnings from `pre-commit` (e.g., run `ruff --fix`)
    - Stage any new fixes and re-run `pre-commit`

3. Commit changes

    - Commit and sync changes
    - Confirm project builds on github with no errors (see 'Actions' tab)
    - Confirm adequate coverage via `codecov` (click link on github)

4. Document changes

    - Edit `CHANGELOG.md` and `README.md` to reflect changes, then commit
    - Edit the version in `setup.py`
    - Tag the release using `git tag -a vX.X.X` (updating `X`s) with a short changelog summary as the tag message
    - Push changes `git push --tags`

5. Build the release

    - Build using `python -m build `
    - Add a release to the [github page](https://github.com/GallowayLabMIT/rushd/releases) by copy-pasting the changelog
    - Add the `.whl` and `.tar.gz` files (from the `dist` folder, delete old versions) to the github release
    - Upload the package to PyPI using `twine upload dist/*`. Note that this requires a PyPI API token from a project collaborator,
      one of which is the Galloway Lab account (see password database for login).


## Changelog
See the [CHANGELOG](CHANGELOG.md) for detailed changes.
```
## [0.7.0] - 2026-02-26
### Modified
- The following flow functions can now take zip files, not just folders.
  This speeds up data loading especially on networked filesystems.
  - `rd.flow.load_csv`
  - `rd.flow.load_csv_with_metadata`
  - `rd.flow.load_groups_with_metadata`
````

## License
This is licensed by the MIT license. Use freely!

## What does the name mean?
The name is a reference to [Ibn Rushd](https://en.wikipedia.org/wiki/Averroes), a Muslim scholar born in Córdoba who was responsible for translating and adding scholastic commentary to ancient Greek works, especially Aristotle. His translations spurred further translations into Latin and Hebrew, reigniting interest in ancient Greek works for the first time since the fall of the Roman empire.

His name is pronounced [rush-id](https://translate.google.com/?sl=auto&tl=en&text=%20%D8%A7%D8%A8%D9%86%20%D8%B1%D8%B4%D8%AF&op=translate).

If we take the first and last letter, we also get `rd`: repeatable data!

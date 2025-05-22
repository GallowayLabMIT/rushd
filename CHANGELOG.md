# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2025-05-22
### Modified
- Switched to using `np.nan` instead of `np.NaN` to be compatible with Numpy 2.0
- Removed support for Python 3.7 and added support for 3.13

## [0.5.0] - 2024-04-15
### Added
- Added new `rd.plot.debug_axes` which draws guide lines to help with axis alignment.
- Added new `rd.plot.adjust_subplot_margins_inches` which allows subplot configuring
  using inch offsets (instead of subfigure coordinate offsets)

### Modified
- `rd.flow.load_csv_with_metadata` and
  `rd.flow.load_groups_with_metadata` can now load a subset of columns.
- The `datadir.txt` can include paths that use `~` to represent the home directory.
- `rd.plot.generate_xticklabels` does not include metadata key labels in plots without yticklabels
- `rd.plot.generate_xticklabels` no longer throws an error when xticklabels don't match the dictionary passed (instead leaves labels as-is)
- `rd.plot.generate_xticklabels` now enables user-specified line spacing

## [0.4.2] - 2023-07-27
### Added
- `rd.plot.generate_xticklabels` replaces a plot's existing xticklabels with specified metadata in a table-like format

## [0.4.1] - 2023-06-27
### Modified
- Updated the `rd.plot.plot_mapping` command to properly handle the single-numeric case.

## [0.4.0] - 2023-04-21
### Added
- `rd.plot.plot_well_metadata` to make nice plots corresponding to well metadata specified as a YAML file.

## [0.3.0] - 2023-03-01
### Added
- `rd.flow.load_groups_with_metadata` loads files from several folders (groups, e.g. corresponding to plates) into a single dataframe
### Modified
- `rd.flow.moi` properly generates plots of MOI vs fraction infected for checking calculation accuracy
- `rd.well_mapper` properly handles metadata concatenation

## [0.2.0] - 2022-06-17
### Added
- `rd.flow.moi` calculates viral MOI, creating summary graphs and tables.
### Modified
- `rd.outfile` now creates necessary subdirectories within rootdir/datadir.
- `rd.flow.load_csv_with_metadata` now allows `str` and `Path` arguments
- `rd.flow.load_csv_with_metadata` properly handles well IDs up to A1-P24 (384-well plate)
- `rd.flow.load_csv_with_metadata` fills unspecified metadata with `NA`

## [0.1.0] - 2022-03-11
### Added
- Initial flow processing workflow
### Modified
- Automatic use of black, isort, and flake in CI

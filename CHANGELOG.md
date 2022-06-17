# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

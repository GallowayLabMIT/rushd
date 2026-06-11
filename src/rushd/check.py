"""Helper functions that perform environment sanity checks."""

import importlib.metadata
import json
import shlex
import subprocess
import sys
import tempfile
import warnings
from importlib.metadata import Distribution
from pathlib import Path
from typing import List, Optional, Tuple

from packaging.requirements import Requirement

from .io import NoDatadirError


def in_git_repo() -> bool:
    """Check that we are in a git repo."""
    result = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True)
    return result.returncode == 0


def git_repo_root() -> Path:
    """Return the root of the current repository."""
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True)
    result.check_returncode()

    return Path(result.stdout.decode().strip()).resolve()


def is_running_in_venv() -> bool:
    """Return if the current Python instance is running in a virtual environment."""
    return sys.prefix != sys.base_prefix


def _nb_clean_filter() -> Optional[List[str]]:
    """
    Return command line arguments that run the installed notebook cleaner script.

    Returns None if such a filter does not exist
    """
    result = subprocess.run(["git", "config", "filter.nb-clean.clean"], capture_output=True)
    if result.returncode != 0:
        return None

    return shlex.split(result.stdout.decode())


def _nb_clean_functioning(repo_root: Path, filter_args: List[str]) -> bool:
    """Check if the specified nb-clean call can clean a test notebook."""
    unclean_nb_json = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "18eb8ea1",
                "metadata": {},
                "outputs": [],
                "source": ["import pandas as pd\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "env", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.4",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    clean_nb_json = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "18eb8ea1",
                "metadata": {},
                "outputs": [],
                "source": ["import pandas as pd\n"],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "env", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with tempfile.TemporaryDirectory() as tm:
        unclean_path = (Path(tm) / "unclear.ipynb").resolve()
        with unclean_path.open("w") as file:
            json.dump(unclean_nb_json, file)
        # run the notebook cleaner
        _ = subprocess.run(filter_args + [str(unclean_path)], cwd=repo_root, capture_output=True)

        with unclean_path.open("r") as file:
            cleaned_json = json.load(file)
        return cleaned_json == clean_nb_json


def _datadir_is_gitignored(repo_root: Path, rd_rootdir: Path) -> bool:
    """Return true if the datadir.txt file is gitignored."""
    # construct the correct relative path from the repo root to the datadir.txt file
    datadir_path = rd_rootdir / "datadir.txt"

    datadir_relpath = datadir_path.resolve().relative_to(repo_root)
    result = subprocess.run(
        ["git", "check-ignore", str(datadir_relpath)], cwd=repo_root, capture_output=True
    )
    return result.returncode == 0


def _locate_requirements() -> Optional[Path]:
    """Locate the requirements.txt using datadir.txt or git to find the local root."""
    try:
        from .io import rootdir

        requirements_path = rootdir / "requirements.txt"
    except NoDatadirError:
        if not in_git_repo():
            return None
        repo_root = git_repo_root()
        requirements_path = repo_root / "requirements.txt"

    if requirements_path.exists():
        return requirements_path

    return None


def _diff_requirements(
    requirements: List[Requirement], installed: List[Distribution]
) -> List[Tuple[Requirement, Distribution]]:
    """
    Check that all current requirements are a superset of tracked requirements.

    Returns a list of requirement/distribution pairs where the installed package does not match
    the given requirements

    Validates that key packages appear in the file_req list.

    Raises warnings for any problems, returning True if there are any problems
    """
    mismatches = []

    req_lookup = {r.name: r for r in requirements}

    for package in installed:
        if package.name in req_lookup:
            req = req_lookup[package.name]
            if not req.specifier.contains(package.version):
                mismatches.append((req, package))
    return mismatches


def _ensure_packages_present(
    requirements: List[Requirement], installed: List[Distribution]
) -> List[Distribution]:
    """
    Check that certain key packages are listed in the requirements list, if present.

    Returns the list of packages that are not present in requirements
    """
    key_packages = ["rushd", "numpy", "scipy", "pandas", "seaborn", "matplotlib", "pyarrow"]

    missing_reqs = []
    req_lookup = {r.name: r for r in requirements}

    installed_key_packages = [p for p in installed if p.name in key_packages]
    for package in installed_key_packages:
        if package.name not in req_lookup:
            missing_reqs.append(package)
    return missing_reqs


def sanity_check_venv() -> bool:
    """Perform virtual environment sanity check, returning false if not running in a venv."""
    if not is_running_in_venv():
        warnings.warn("Python is not running inside a virtual environment!", stacklevel=2)
        return False
    return True


def sanity_check_requirements() -> bool:
    """Check that a requirements.txt file exists and is up to date."""
    # check for a requirements.txt file.
    requirements_path = _locate_requirements()

    if requirements_path is None:
        warnings.warn(
            "No requirements.txt file accessible! Expected pinned dependencies stored at "
            + f"{str(requirements_path)}",
            stacklevel=2,
        )
        # return early if we couldn't locate the requirements.txt file
        return False

    # check if the requirements are out of date.
    # out of date if:
    # 1. the installed version of a package doesn't match the version in requirements.txt
    # 2. if a key package is missing from requirements.txt
    sane = True

    # check for superset of requirements
    installed_packages = list(importlib.metadata.distributions())
    requirements = [Requirement(r) for r in requirements_path.read_text().strip().split("\n")]

    diffed_packages = _diff_requirements(requirements, installed_packages)
    missing_packages = _ensure_packages_present(requirements, installed_packages)

    if len(diffed_packages) > 0 or len(missing_packages) > 0:
        sane = False

    for req, package in diffed_packages:
        warnings.warn(
            f"Package {package.name} has version {package.version} installed, "
            + f"but has version {req.specifier} in requirements.txt file! "
            + "Repeat `pip freeze > requirements.txt`",
            stacklevel=2,
        )
    for package in missing_packages:
        warnings.warn(
            f"Package {package.name} is installed but not listed in requirements.txt!"
            + " Repeat `pip freeze > requirements.txt`",
            stacklevel=2,
        )

    return sane


def sanity_check_git() -> bool:
    """Check if files are properly git tracked."""
    if not in_git_repo():
        return True

    sane = True

    repo_root = git_repo_root()
    # always check for datadir sanity
    try:
        from .io import rootdir

        if not _datadir_is_gitignored(repo_root, rootdir):
            warnings.warn("datadir.txt file is not git-ignored! Add to .gitignore", stacklevel=2)
            sane = False
    except NoDatadirError:
        # if there's no datadir, not our problem
        pass

    # check that requirements.txt is tracked
    requirements_path = _locate_requirements()

    if requirements_path is not None:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                str(requirements_path.relative_to(repo_root)),
            ],
            cwd=repo_root,
            capture_output=True,
        )
        print(
            f"Requirements git check: return: {result.returncode}"
            + f"\nstdout:\n{result.stdout.decode()}"
            + f"\nstderr:\n{result.stderr.decode()}"
        )
        if result.returncode != 0:
            warnings.warn(
                "The requirements.txt file exists but is not tracked by git!", stacklevel=2
            )
            sane = False
    return sane


def sanity_check_nb_clean() -> bool:
    """Check if nb-clean is properly installed."""
    if not in_git_repo():
        return True

    sane = True

    repo_root = git_repo_root()
    filter_args = _nb_clean_filter()
    if filter_args is None:
        warnings.warn("No nb-clean filter set! Run nb-clean add-filter", stacklevel=2)
        sane = False
    else:
        if not _nb_clean_functioning(repo_root, filter_args):
            warnings.warn("nb-clean failed to clean test file!", stacklevel=2)
            sane = False

    return sane


def sanity_check() -> bool:
    """
    Perform various sanity checks on the loaded environment.

    Returns true if the environment is sane.
    """
    sanity_checks = [
        sanity_check_venv,
        sanity_check_requirements,
        sanity_check_git,
        sanity_check_nb_clean,
    ]
    check_results = [f() for f in sanity_checks]

    return all(check_results)

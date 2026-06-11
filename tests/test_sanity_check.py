import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

import rushd


def test_venv_detection(monkeypatch):
    """Tests that virtual environments are detected properly"""
    with warnings.catch_warnings(record=True) as w:
        monkeypatch.setattr(sys, "base_prefix", sys.prefix)
        assert not rushd.check.sanity_check_venv()
        assert len(w) == 1
        assert "virtual environment" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        monkeypatch.setattr(sys, "base_prefix", sys.prefix + "hi")
        assert rushd.check.sanity_check_venv()
        assert len(w) == 0


@pytest.fixture(scope="class")
def class_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("class_scope")


@pytest.fixture(scope="class")
def git_repo(class_tmp: Path) -> None:
    """Sets up a base Git repository"""
    subprocess.run(["git", "init"], cwd=class_tmp, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "initial"],
        cwd=class_tmp,
        check=True,
        capture_output=True,
    )


@dataclass
class VirtualEnvironment:
    python: Path
    pip: Path
    root: Path


@pytest.fixture(scope="class")
def venv(class_tmp: Path, request) -> VirtualEnvironment:
    """Sets up a basic Python virtual environment with rushd installed. Returns the python path"""
    subprocess.run(["python", "-m", "venv", "env"], cwd=class_tmp, check=True, capture_output=True)
    python_path = class_tmp.resolve() / "env" / "bin" / "python"
    pip_path = class_tmp.resolve() / "env" / "bin" / "pip"
    subprocess.run(
        [str(pip_path), "install", "nb-clean", "coverage"], cwd=class_tmp, capture_output=True
    )
    subprocess.run(
        [str(pip_path), "install", "-e", request.path.parent.parent],
        cwd=class_tmp,
        capture_output=True,
    )

    return VirtualEnvironment(python=python_path, pip=pip_path, root=class_tmp)


def run_pip(venv: VirtualEnvironment, *args) -> subprocess.CompletedProcess:
    """Runs a pip command in the virtual environment"""
    return subprocess.run([str(venv.pip), *args], cwd=venv.root, capture_output=True, check=True)


def run_python_with_cov(venv: VirtualEnvironment, lines: List[str]) -> subprocess.CompletedProcess:
    """Runs a series of python commands, loading coverage first"""
    pre_lines = ["import coverage", "coverage.process_startup()"]

    return subprocess.run(
        [str(venv.python), "-c", "; ".join(pre_lines + lines)], cwd=venv.root, capture_output=True
    )


def run_sanity_check(venv: VirtualEnvironment) -> subprocess.CompletedProcess:
    """Runs the full rushd sanity check"""
    return run_python_with_cov(venv, ["import rushd", "rushd.check.sanity_check()"])


@pytest.mark.usefixtures("venv", "class_tmp")
class TestSanityCheckOutsideGit:
    def test_no_git_repo(self, class_tmp: Path, venv):
        """Test if the sanity check works outside of a git repo"""

        result = run_sanity_check(venv)
        assert "No requirements.txt file accessible!" in result.stderr.decode()

        # set a datadir since we are outside of a git repo; we still shouldn't find
        # a requirements.txt file
        (class_tmp / "datadir.txt").write_text(str(class_tmp))

        result = run_sanity_check(venv)
        assert "No requirements.txt file accessible!" in result.stderr.decode()

        result = run_pip(venv, "list", "--format=freeze")

        (class_tmp / "requirements.txt").write_text(result.stdout.decode())
        result = run_sanity_check(venv)
        (class_tmp / "requirements.txt").unlink()

        assert "No requirements.txt file accessible!" not in result.stderr.decode()


@pytest.mark.usefixtures("git_repo", "venv", "class_tmp")
class TestSanityCheck:
    def test_in_venv(self, class_tmp: Path, venv):
        """Tests that we are properly in a virtual environment"""
        result = run_python_with_cov(venv, ["import rushd", "rushd.check.sanity_check_venv()"])

        assert "virtual environment" not in result.stderr.decode()

    def test_non_ignored_datadir(self, class_tmp: Path, venv):
        """Tests that an existing datadir.txt file that is not git-ignored gives a warning"""
        (class_tmp / "datadir.txt").write_text(str(class_tmp))

        result = run_sanity_check(venv)

        (class_tmp / "datadir.txt").unlink()
        assert "datadir.txt file is not git-ignored" in result.stderr.decode()

    def test_ignored_datadir(self, class_tmp: Path, venv):
        """Tests that an existing datadir.txt file that is git-ignored does not give a warning"""
        (class_tmp / "datadir.txt").write_text(str(class_tmp))
        (class_tmp / ".gitignore").write_text("datadir.txt")

        result = run_sanity_check(venv)

        (class_tmp / "datadir.txt").unlink()
        (class_tmp / ".gitignore").unlink()
        assert "datadir.txt file is not git-ignored" not in result.stderr.decode()

    def test_nb_clean_uninstalled(self, class_tmp: Path, venv):
        """Test that nb-clean warning is thrown when nb-clean is not installed"""
        result = run_sanity_check(venv)
        assert "No nb-clean filter set!" in result.stderr.decode()

    def test_nb_clean_installed(self, class_tmp: Path, venv, git_repo):
        """Test that properly-installed nb-clean doesn't throw a warning"""
        _ = subprocess.run(
            [
                "git",
                "config",
                "filter.nb-clean.clean",
                str(class_tmp / "env" / "bin" / "nb-clean"),
                "clean",
            ],
            cwd=class_tmp,
            capture_output=True,
            check=True,
        )

        result = run_sanity_check(venv)

        _ = subprocess.run(
            ["git", "config", "--unset", "filter.nb-clean.clean"],
            cwd=class_tmp,
            capture_output=True,
            check=True,
        )
        assert "No nb-clean filter set!" not in result.stderr.decode()

    def test_no_requirements_file(self, class_tmp: Path, venv):
        """Test that a warning is given if there is no requirements.txt file"""
        result = run_sanity_check(venv)
        assert "No requirements.txt file accessible!" in result.stderr.decode()

    def test_untracked_requirements_file(self, class_tmp: Path, venv):
        """Test that a warning is given when the requirements.txt file is untracked"""
        result = run_pip(venv, "list", "--format=freeze")

        (class_tmp / "requirements.txt").write_text(result.stdout.decode())
        result = run_sanity_check(venv)
        (class_tmp / "requirements.txt").unlink()

        assert "requirements.txt file exists but is not tracked" in result.stderr.decode()

    def test_tracked_requirements_file(self, class_tmp: Path, venv):
        """Test that a properly tracked requirements file doesn't give a warning"""
        result = run_pip(venv, "list", "--format=freeze")
        (class_tmp / "requirements.txt").write_text(result.stdout.decode())
        _ = subprocess.run(
            ["git", "add", "requirements.txt"], cwd=class_tmp, capture_output=True, check=True
        )
        result = run_sanity_check(venv)

        _ = subprocess.run(
            ["git", "reset", "HEAD", "--", "requirements.txt"],
            cwd=class_tmp,
            capture_output=True,
            check=True,
        )
        (class_tmp / "requirements.txt").unlink()
        assert "requirements.txt file exists but is not tracked" not in result.stderr.decode()

    def test_requirements_file_subset(self, class_tmp: Path, venv):
        """
        Test for strict requirement subset.

        Raises a warning if the requirements file is not a superset
        of the installed packages
        """
        result = run_pip(venv, "list", "--format=freeze")

        # adjust written requirements file
        req_lines = result.stdout.decode().strip().split("\n")
        req_lines = [line for line in req_lines if "rushd" not in line] + ["rushd==0.1"]
        (class_tmp / "requirements.txt").write_text("\n".join(req_lines))

        result = run_sanity_check(venv)

        (class_tmp / "requirements.txt").unlink()
        assert "Package rushd has version" in result.stderr.decode()

    def test_key_package(self, class_tmp: Path, venv):
        """Test that a warning is given if the requirements file does not include a key package"""
        result = run_pip(venv, "list", "--format=freeze")

        # adjust written requirements file
        req_lines = result.stdout.decode().strip().split("\n")
        req_lines = [line for line in req_lines if "rushd" not in line]
        (class_tmp / "requirements.txt").write_text("\n".join(req_lines))

        result = run_sanity_check(venv)

        (class_tmp / "requirements.txt").unlink()
        assert (
            "Package rushd is installed but not listed in requirements.txt"
            in result.stderr.decode()
        )

import datetime
import hashlib
import os
import subprocess
import sys
from importlib import reload
from pathlib import Path

import pytest
import rushd
import rushd.io
import yaml


def test_datadir_rootdir(tmp_path: Path):
    """
    Tests datadir/rootdir discovery, for
    a datadir.txt in the same directory and in
    a parent directory.
    """
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "inner").mkdir()
    (tmp_path / "data").mkdir()
    with (tmp_path / "root" / "datadir.txt").open("w") as datadir_txt:
        datadir_txt.write(str(tmp_path / "data"))

    # Test same-directory datadir.txt lookup
    os.chdir(tmp_path / "root")
    reload(rushd.io)
    assert rushd.datadir == (tmp_path / "data")
    assert rushd.rootdir == (tmp_path / "root")

    # Test parent-directory datadir.txt lookup
    os.chdir(tmp_path / "root" / "inner")
    reload(rushd.io)
    assert rushd.datadir == (tmp_path / "data")
    assert rushd.rootdir == (tmp_path / "root")


def test_subdirectory_outdir_creation(tmp_path: Path):
    """
    Tests that rd.io.outfile creates needed subdirectories
    when creating an outfile, but only if the path is relative
    to the rootdir or datadir.
    """
    (tmp_path / "root").mkdir()
    (tmp_path / "data").mkdir()
    with (tmp_path / "root" / "datadir.txt").open("w") as datadir_txt:
        datadir_txt.write(str(tmp_path / "data"))
    os.chdir(tmp_path / "root")
    reload(rushd.io)

    _ = rushd.outfile(tmp_path / "root" / "inner" / "inner_test.txt")
    _ = rushd.outfile(tmp_path / "data" / "even" / "more" / "nesting" / "inner_test.txt")

    assert (tmp_path / "root" / "inner").exists()
    assert (tmp_path / "data" / "even" / "more" / "nesting").exists()

    # Try test case where we are _outside_ the root or data
    with pytest.raises(FileNotFoundError):
        _ = rushd.outfile(tmp_path / "somewhere" / "else" / "external_text.txt")


def test_extra_whitespace_datadir(tmp_path: Path):
    """
    Tests that extra (newline) whitespace is removed
    before processing the datadir and rootdir.
    """
    (tmp_path / "root").mkdir()
    (tmp_path / "data").mkdir()
    with (tmp_path / "root" / "datadir.txt").open("w") as datadir_txt:
        datadir_txt.write("\n" + str(tmp_path / "data") + "\n")
    # Test same-directory datadir.txt lookup
    os.chdir(tmp_path / "root")
    reload(rushd.io)
    assert rushd.datadir == (tmp_path / "data")
    assert rushd.rootdir == (tmp_path / "root")


def test_datadir_rootdir_failure(tmp_path: Path):
    """
    Tests that failed datadir/rootdir discovery leads to
    the proper exception
    """
    os.chdir(tmp_path)
    reload(rushd.io)
    with pytest.raises(rushd.io.NoDatadirError):
        _ = rushd.datadir
    with pytest.raises(rushd.io.NoDatadirError):
        _ = rushd.rootdir


def test_git_version(tmp_path: Path):
    """
    Tests that git repositories are properly
    recognized. Three subcases are handled:
    1. Clean git repository
    2. Unclean git repository
    3. Not a git repository
    """
    # Test that git is runnable
    git_result = subprocess.run(["git", "--version"], check=False, capture_output=True)
    if git_result.returncode != 0:
        pytest.skip("No git support")

    (tmp_path / "clean_repo").mkdir()
    (tmp_path / "dirty_repo").mkdir()
    (tmp_path / "not_a_repo").mkdir()

    os.chdir(tmp_path / "clean_repo")
    subprocess.run(["git", "init"], check=True, capture_output=True)
    with (tmp_path / "clean_repo" / "test.txt").open("w") as f:
        f.write("Hello world")
    subprocess.run(["git", "add", "test.txt"], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--author", "CI <>", "-m", "initial"],
        check=True,
        capture_output=True,
    )

    clean_version = rushd.io.git_version()
    assert clean_version is not None
    assert not clean_version.endswith("dirty")

    os.chdir(tmp_path / "dirty_repo")
    subprocess.run(["git", "init"], check=True, capture_output=True)
    with (tmp_path / "dirty_repo" / "test.txt").open("w") as f:
        f.write("Hello world")
    subprocess.run(["git", "add", "test.txt"], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--author", "CI <>", "-m", "initial"],
        check=True,
        capture_output=True,
    )
    with (tmp_path / "dirty_repo" / "test.txt").open("w") as f:
        f.write("Hello world...again")

    dirty_version = rushd.io.git_version()
    assert dirty_version is not None
    assert dirty_version.endswith("dirty")

    os.chdir(tmp_path / "not_a_repo")
    assert rushd.io.git_version() is None


def test_infile_outfile_no_datadir(tmp_path: Path):
    """
    Tests that in/out files can be written
    and edited without needing a data directory
    """
    with (tmp_path / "foo.txt").open("w") as f:
        f.write("Hello world")
    _ = rushd.io.infile(tmp_path / "foo.txt")
    with rushd.io.outfile(tmp_path / "bar.txt").open("w") as f:
        f.write("Goodbye world")
    with (tmp_path / "bar.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert len(meta["dependencies"]) == 1


def test_path_translation(tmp_path: Path):
    """
    Tests that infiles can be loaded
    relative to the data directory, the root directory,
    and via an absolute path
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "root").mkdir()
    (tmp_path / "external").mkdir()
    with (tmp_path / "root" / "datadir.txt").open("w") as datadir_txt:
        datadir_txt.write(str(tmp_path / "data"))

    with (tmp_path / "data" / "in_data.txt").open("w") as f:
        f.write("foo")
    with (tmp_path / "root" / "in_root.txt").open("w") as f:
        f.write("bar")
    with (tmp_path / "external" / "in_external.txt").open("w") as f:
        f.write("baz")
    os.chdir(tmp_path / "root")
    reload(rushd.io)
    # Generate an output file and check the relative paths created
    _ = rushd.io.infile(tmp_path / "data" / "in_data.txt")
    _ = rushd.io.infile(tmp_path / "root" / "in_root.txt")
    _ = rushd.io.infile(tmp_path / "external" / "in_external.txt")
    with rushd.io.outfile(tmp_path / "root" / "out.txt").open("w") as f:
        f.write("Hello world!")

    with (tmp_path / "root" / "out.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert len(meta["dependencies"]) == 3
        for file_dict in meta["dependencies"]:
            filename = file_dict["file"]
            path_type = file_dict["path_type"]
            if filename.endswith("in_data.txt"):
                assert Path(filename) == Path("in_data.txt")
                assert path_type == "datadir_relative"
            if filename.endswith("in_root.txt"):
                assert Path(filename) == Path("in_root.txt")
                assert path_type == "rootdir_relative"
            if filename.endswith("in_external.txt"):
                assert Path(filename) == (tmp_path / "external" / "in_external.txt").resolve()
                assert path_type == "absolute"


def test_outfile_properties(tmp_path: Path):
    """
    Tests that the other outfile properties are
    properly written
    """
    os.chdir(tmp_path)
    reload(rushd.io)
    with (tmp_path / "in.txt").open("w") as f:
        f.write("hashbrowns")
    _ = rushd.io.infile(tmp_path / "in.txt")
    with rushd.io.outfile(tmp_path / "out.txt").open("w") as f:
        f.write("Hello world!")
    with (tmp_path / "out.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert meta["name"] == "out.txt"
        assert meta["type"] == "tracked_outfile"
        assert meta["date"] == str(datetime.datetime.now().date().isoformat())
        assert Path(meta["dependencies"][0]["file"]) == (tmp_path / "in.txt").resolve()

        sha = hashlib.sha256()
        sha.update("hashbrowns".encode("utf-8"))
        assert meta["dependencies"][0]["sha256"] == sha.hexdigest()


def test_git_property(tmp_path: Path):
    """
    Tests that the git repo status
    property is properly written.
    """
    git_result = subprocess.run(["git", "--version"], check=False, capture_output=True)
    if git_result.returncode != 0:
        pytest.skip("No git support")

    os.chdir(tmp_path)
    reload(rushd.io)
    subprocess.run(["git", "init"], check=True, capture_output=True)
    with (tmp_path / "text.txt").open("w") as f:
        f.write("Hello world!")
    subprocess.run(["git", "add", "text.txt"], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--author", "CI <>", "-m", "initial"],
        check=True,
        capture_output=True,
    )
    git_log = subprocess.run(
        ["git", "log", "-n1", "--format=format:%H"], check=True, capture_output=True
    )

    with rushd.io.outfile(tmp_path / "out.txt").open("w") as f:
        f.write("Hello world!")
    with (tmp_path / "out.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert meta["git_version"] == git_log.stdout.decode()


def test_tagged_io(tmp_path: Path):
    with (tmp_path / "foo_tag_1.txt").open("w") as f:
        f.write("foo")
    with (tmp_path / "bar_tag_1.txt").open("w") as f:
        f.write("bar")
    with (tmp_path / "both_tag_1.txt").open("w") as f:
        f.write("bar")
    _ = rushd.io.infile(tmp_path / "foo_tag_1.txt", "foo")
    _ = rushd.io.infile(tmp_path / "bar_tag_1.txt", "bar")
    _ = rushd.io.infile(tmp_path / "both_tag_1.txt", "foo")
    _ = rushd.io.infile(tmp_path / "both_tag_1.txt", "bar")
    with rushd.io.outfile(tmp_path / "foo_out.txt", "foo").open("w") as f:
        f.write("Hello world!")
    with rushd.io.outfile(tmp_path / "bar_out.txt", "bar").open("w") as f:
        f.write("Hello world!")
    # Test outputs
    with (tmp_path / "foo_out.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert meta["dependencies"][0]["file"].endswith("foo_tag_1.txt")
        assert meta["dependencies"][1]["file"].endswith("both_tag_1.txt")
    with (tmp_path / "bar_out.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert meta["dependencies"][0]["file"].endswith("bar_tag_1.txt")
        assert meta["dependencies"][1]["file"].endswith("both_tag_1.txt")


def test_nohash_infile(tmp_path: Path):
    """
    Tests that infiles are still valid
    if they are not hashed
    """
    os.chdir(tmp_path)
    reload(rushd.io)
    with (tmp_path / "in.txt").open("w") as f:
        f.write("foo")
    _ = rushd.io.infile(tmp_path / "in.txt", should_hash=False)
    with rushd.io.outfile(tmp_path / "out.txt").open("w") as f:
        f.write("Hello world!")
    with (tmp_path / "out.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert "sha256" not in meta["dependencies"][0]


def test_str_in_out(tmp_path: Path):
    os.chdir(tmp_path)
    reload(rushd.io)
    with (tmp_path / "in.txt").open("w") as f:
        f.write("foo")
    _ = rushd.io.infile(str(tmp_path / "in.txt"))
    with rushd.io.outfile(str(tmp_path / "out.txt")).open("w") as f:
        f.write("Hello world!")
    with (tmp_path / "out.txt.yaml").open() as f:
        meta = yaml.safe_load(f)
        assert meta["dependencies"][0]["file"].endswith("in.txt")


def test_permission_denied(tmp_path: Path):
    """
    Makes sure that we correctly handle the
    case where we don't have permission to continue
    upward in the filesystem tree.
    """
    if sys.platform.startswith("win"):
        pytest.skip("Unable to modify file permissions on Windows")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "again").mkdir()
    with (tmp_path / "nested" / "datadir.txt").open("w") as f:
        f.write("access denied here!")
    os.chdir(tmp_path / "nested" / "again")
    os.chmod(tmp_path / "nested", 0o000)
    try:
        reload(rushd.io)
        os.chmod(tmp_path / "nested", 0o700)
        reload(rushd.io)
    finally:
        os.chmod(tmp_path / "nested", 0o700)

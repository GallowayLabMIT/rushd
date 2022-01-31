"""
A submodule implementing common IO handling mechanisms.
## Rationale
File and folder management is a common problem when
handling large datasets. You often want to separate
out large data from your code. How do you keep track
of where your data is, especially if moving between
different computers/clusters?

`rushd.io` adds convenience functions to handle
common cases, as well as writing metadata with
your output files that identify input files.
"""
import datetime
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import warnings
import yaml

## Datadir/rootdir-related detection
class NoDatadirError(RuntimeError):
    """
    Error raised when rushd is unable to locate
    a datadir.txt path in the current file.
    """

def _locate_datadir_txt()->Optional[Path]:
    """
    Starting at the current working directory, walks up the
    filesystem until a root 'datadir.txt' file is found. This
    path is returned.

    Returns
    -------
    A Path pointing to the root datadir file, None if a root datadir could not be found.
    Raises
    ------
    UntrackedRepositoryError: if the search path could not find a root tame.yaml file.
    """
    r_path = Path.cwd()
    try:
        if r_path.is_dir():
            current_dir = r_path
        else:
            current_dir = r_path.parent
        # Resolve to an absolute path
        current_dir = current_dir.resolve()
        while not (current_dir / 'datadir.txt').is_file():
            up_dir = current_dir.parent
            # Make sure we didn't reach the filesystem root
            if up_dir == current_dir:
                return None
            # otherwise, continue searching
            current_dir = up_dir
        return current_dir / 'datadir.txt'
    except PermissionError:
        return None

def _load_root_datadir(datadir_txt: Optional[Path]) -> Tuple[Optional[Path],Optional[Path]]:
    """
    Uses the location of the datadir.txt file to define the data directory
    and the "root" directory.

    Parameters
    ----------
    datadir_txt: The location of the datadir.txt file, if loaded.

    Returns
    -------
    A tuple containing (rootdir, datadir). Each tuple component is None
    if the desired directory is not a accessible directory.
    """
    if datadir_txt is None:
        return (None,None)
    rootdir = datadir_txt.parent
    datadir = Path(datadir_txt.read_text())
    return (
        rootdir if rootdir.is_dir() else None,
        datadir if datadir.is_dir() else None
    )

_rootdir, _datadir = _load_root_datadir(_locate_datadir_txt())
if _datadir is None:
    warnings.warn("Unable to locate datadir.txt", category=ImportWarning)

def __getattr__(name: str) -> Path:
    """
    Sets up module exports.

    rushd.io exports two attributes,
    the datadir export and the rootdir export.
    """
    if name == 'datadir':
        if _datadir is None:
            raise NoDatadirError(f"No datadir.txt file found in working directory {Path.cwd()} or parents")
        return _datadir
    if name == 'rootdir':
        if _rootdir is None:
            raise NoDatadirError(
                f"No datadir.txt file found in working directory {Path.cwd()} or parents, so could not define root")
        return _rootdir
    raise AttributeError(f"No attribute {name} in rushd.io")

def git_version()->Optional[str]:
    """
    Returns the current repository state as a string.
    The state is a string {hash}, with {-dirty} appended
    if there are edits that have not been saved.
    Returns None if the current working directory is
    not contained within a git repository.
    """
    git_log = subprocess.run(['git', 'log', '-n1', '--format=format:%H'], check=False, capture_output=True)
    git_diff_index = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD', '--'], check=False, capture_output=True)

    # Unable to locate git, or not in a repo
    if git_log.returncode != 0:
        return None
    return git_log.stdout.decode() + ('-dirty' if git_diff_index.returncode != 0 else '')

## Convenience functions for storing files and their hashes
_untagged_inputs: Dict[Path,Optional[str]] = {}
_tagged_inputs: Dict[str,Dict[Path,Optional[str]]] = {}

def _is_relative_to(path: Path, base_path: Path) -> bool:
    """
    Checks that a path can be written relative
    to a base path. Needed on Pythons < 3.9.

    Parameters
    ----------
    path: Path
        The path to compare against a base path.
    base_path: Path
        The path that acts as the base, to write `path` relative to.

    Returns
    -------
    True if `path` can be written as a relative path to `base_path`, False otherwise
    """
    try:
        _ = path.relative_to(base_path)
        return True
    except ValueError:
        return False

def infile(filename: Union[str,Path], tag: Optional[str] = None, should_hash: bool = True) -> Path:
    """
    Passthrough wrapper around a path that (optionally)
    hashes and adds the file to a internally tracked list.
    This list accumulates files that potentially went into
    creation of an output file.

    Parameters
    ----------
    filename: str or Path
        The filename of the input file to open.
    tag: str (optional)
        A user-defined tag that organizes opened files.
    should_hash: bool
        If the input file should be hashed. You may want to skip
        this if the file is extremely large.

    Returns
    -------
    A Path object that represents the same file as `filename`.
    """
    if not isinstance(filename, Path):
        filename = Path(filename)
    # Hash the file
    if should_hash:
        chunk_size = 2**20
        sha256 = hashlib.sha256()
        with open(filename, 'rb') as bfile:
            while True:
                data = bfile.read(chunk_size)
                if not data:
                    break
                sha256.update(data)
        hash_result = sha256.hexdigest()
    else:
        hash_result = None

    if tag:
        if tag not in _tagged_inputs:
            _tagged_inputs[tag] = {}
        _tagged_inputs[tag][filename] = hash_result
    else:
        _untagged_inputs[filename] = hash_result
    return filename

def outfile(filename: Union[str,Path], tag: Optional[str] = None) -> Path:
    """
    Passthrough method that write a YAML file defining
    which files went into creating a certain output file.

    Parameters
    ----------
    filename: str or Path
        An output filename to write data to.
    tag: str
        A user-defined string that groups input and output files together.

    Returns
    -------
    A Path object that represents the same file as `filename`.

    Side-effects
    ------------
    For output file `out.txt`, writes a YAML file `out.txt.yaml`
    that encodes the following type of metadata:

    ```yaml
    type: tracked_outfile
    name: out.txt
    date: 2022-01-31
    git_version: 13a81aa2a7b1035f6b59c2323b0a7c457eb1657e
    dependencies:
      - file: some_infile.csv
        path_type: datadir_relative
    ```
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    yaml_result: Dict[str,Union[str,List[Dict[str,str]]]] = {
        'type': 'tracked_outfile',
        'name': filename.name,
        'date': datetime.datetime.now().date().isoformat(),
    }
    # Save git version if we are in a git repo
    git = git_version()
    if git:
        yaml_result['git_version'] = git

    if tag:
        files: Dict[Path, Optional[str]] = _tagged_inputs[tag] if tag in _tagged_inputs else {}
    else:
        files: Dict[Path, Optional[str]] = _untagged_inputs
    file_yaml: List[Dict[str,str]] = []
    for filepath, file_hash in files.items():
        result: Dict[str,str] = {}
        abs_filepath = filepath.resolve()
        abs_datadir = _datadir.resolve() if _datadir else None
        abs_rootdir = _rootdir.resolve() if _rootdir else None

        if abs_datadir and _is_relative_to(abs_filepath, abs_datadir):
            result.update({
                'file': str(abs_filepath.relative_to(abs_datadir)),
                'path_type': 'datadir_relative'
            })
        elif abs_rootdir and _is_relative_to(abs_filepath, abs_rootdir):
            result.update({
                'file': str(abs_filepath.relative_to(abs_rootdir)),
                'path_type': 'rootdir_relative'
            })
        else:
            result.update({
                'file': str(abs_filepath),
                'path_type': 'absolute'
            })
        if file_hash:
            result.update({
                'sha256': file_hash
            })
        file_yaml.append(result)
    yaml_result.update({'dependencies': file_yaml})

    with (filename.parent / (filename.name + '.yaml')).open('w') as yaml_out:
        yaml.dump(yaml_result, yaml_out) # type: ignore
    return filename

## Convenience function for caching dataframes
def cache_dataframe():
    pass
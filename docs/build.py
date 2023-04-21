"""Helper build script that calls apidoc and then sphinx."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Generates HTML documentation")
parser.add_argument("--force-rebuild", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    # Calculate docs path:
    source_path = Path(__file__).resolve().parent.parent / "src"
    docs_path = Path(__file__).resolve().parent
    output_path = Path(__file__).resolve().parent / "_build"
    # Remove the output folder if it exists and we force a rebuild
    if args.force_rebuild and output_path.is_dir():
        shutil.rmtree(output_path)
    python_exe = sys.executable

    # Run build code scripts
    subprocess.run(
        [python_exe, docs_path / "_build_code" / "plot_well_metadata.py"],
        cwd=docs_path / "_build_code",
    )

    autodoc_args = [
        python_exe,
        "-m",
        "sphinx.ext.apidoc",
        "-f",
        "-o",
        str(docs_path / "api"),
        str(source_path / "rushd"),
    ]
    html_args = [
        python_exe,
        "-m",
        "sphinx.cmd.build",
        "-b",
        "html",
        str(docs_path),
        str(output_path),
    ]
    subprocess.run(autodoc_args)
    # Remove
    if (docs_path / "api" / "modules.rst").exists():
        (docs_path / "api" / "modules.rst").unlink()
    subprocess.run(html_args)

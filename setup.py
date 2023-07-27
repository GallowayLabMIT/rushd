"""Setup configuration rushd.

For local development, use
`pip install -e .[dev]`
which will install additional dev tools.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rushd",
    version="0.4.2",
    author="Christopher Johnstone, Kasey Love, Conrad Oakes",
    author_email="meson800@gmail.com",
    description="Package for maintaining robust, reproducible data management.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GallowayLabMIT/rushd",
    project_urls={
        "Bug Tracker": "https://github.com/GallowayLabMIT/rushd/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "pyyaml",
        "pandas",
        "matplotlib",
        "scipy",
        'typing-extensions;python_version<"3.8"',
    ],
    extras_require={
        "dev": [
            "pyarrow",
            "pytest",
            "pytest-pep8",
            "pytest-cov",
            "pytest-mock",
            "pre-commit",
            "ruff",
            "build",
            "twine",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
        ]
    },
)

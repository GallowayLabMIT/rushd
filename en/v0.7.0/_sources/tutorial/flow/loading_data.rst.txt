==================
Loading flow data
==================

``rushd`` provides several convenient ways to load flow data.
We recommend that you put all metadata into a YAML file and load it based
on auto-generated well-ID information, but you can also load metadata
directly from CSV filenames.

Metadata in YAML
----------------
You can load extra metadata using a YAML file that defines
which wells had which conditions / treatments / cell lines / etc.

For example, this YAML files specifies circuit syntax and dox treatment:

.. code-block:: yaml

    metadata:
      syntax:
        - tandem: A1-A12
        - convergent: B1-B12
        - divergent: C1-C12
      dox_ng:
        - 0: A1-C6
        - 1000: A7-C12

You are required to place all metadata conditions inside a top-level
key called ``metadata``. Within this, you can define arbitrary mappings
that map onto ranges of wells.

To load a plate of flow data, you can use ``rushd`` by specifying
a path to this YAML file and a path to the folder containing the ``.csv`` files:

.. code-block:: python

    df = rd.flow.load_csv_with_metadata(rd.datadir/"exp01"/"metadata.yaml", rd.datadir/"exp01"/"csv_export")

Alternatively, you can specify a ``.zip`` file of the metadata. Let's assume you zipped all of the CSVs into
one file, called ``csvs.zip``. Then, you can load this as:

.. code-block:: python

    df = rd.flow.load_csv_with_metadata(rd.datadir/"exp01"/"metadata.yaml", rd.datadir/"exp01"/"csvs.zip")

Finally, you are allowed to zip the metadata and CSVs together. Let's say you have the following zip file:

.. code-block:: text

    exp01.zip/
    ├── metadata.yaml
    └── export/
        ├── export_A1_singlets.csv
        ├── export_A2_singlets.csv
        ├── ...
        └── export_G12_singlets.csv

You can load this by specifying the path to the metadata file and the CSVs as a tuple:

.. code-block:: python

    df = rd.flow.load_csv_with_metadata((rd.datadir/"exp01.zip", "metadata.yaml"), (rd.datadir/"exp01.zip", "export"))

Check out the documentation for this function for more things you can do, like specifying only certain
columns to be loaded:

.. autofunction:: rushd.flow.load_csv_with_metadata
    :noindex:

Finally, you can use any of these data loading techniques with the multi-plate loading function:

.. autofunction:: rushd.flow.load_groups_with_metadata
    :noindex:

Metadata in filenames
---------------------
If you ran a tube experiment or otherwise have metadata specified in filenames, you can use a function that just
loads CSVs and extracts the metadata out of.

Let's say that we have some files that have metadata in their filenames, like:

- ``export_BFP_100_singlets.csv``
- ``export_GFP_1000_singlets.csv``

where we want to extract the construct and the dox concentration. Developing the regex is beyond the scope here:
use https://regex101.com to evaluate teh regex. In this case, a regex that works here is ``^.*export_(?P<construct>.+)_(?P<dox>[0-9]+)_(?P<population>.+)\.csv``

.. code-block:: python

    regex = r"^.*export_(?P<construct>.+)_(?P<dox>[0-9]+)_(?P<population>.+)\.csv"
    df = rd.flow.load_csv(rd.datadir/"exp02", regex)


You can see more details of this function below:

.. autofunction:: rushd.flow.load_csv
    :noindex:

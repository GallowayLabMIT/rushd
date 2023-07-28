==============================
Visualizing YAML well metadata
==============================

Many of the features of `rushd` load metadata using YAML files.
However, writing these files can be error-prone, especially with complicated
experimental setups.

`rushd` also includes plotting functions that can generate well maps automatically for you!
Enter `plot_well_metadata`, which has the following interface:

.. autofunction:: rushd.plot.plot_well_metadata
    :noindex:

Automatic plotting of well metadata
-----------------------------------

Consider the following YAML metadata file:

.. include:: ../_static/example_plot_metadata.yaml
    :code: yaml


By default, `plot_well_metadata` will use a default color scheme for every column
listed in the metadata file. For numerical columns, it will auto-detect if a linear
or log plot is better, and use a Viridis color scheme (with the yellow part of viridis removed).

For the above YAML file, we can get the following plots:

.. include:: ../_build_code/plot_well_metadata.py
    :code: python
    :start-line: 29
    :end-line: 32

.. figure:: ../_static/built_output/plot_metadata_overall/inducible_fluorophore.svg
    :width: 60%
    :align: center

.. figure:: ../_static/built_output/plot_metadata_overall/orientation.svg
    :width: 60%
    :align: center

.. figure:: ../_static/built_output/plot_metadata_overall/inducible_spacerlength.svg
    :width: 60%
    :align: center

.. figure:: ../_static/built_output/plot_metadata_overall/dox.svg
    :width: 60%
    :align: center

Here, you can see the automatic color assignment for both categorical variables, linear numerical variables,
and log-distributed variables.

.. note::

    The automatic detection of log-distributed data examines the median of the values.
    If all values are non-negative and the median is outside of the percentile range
    [0.15, 0.85], the data is detected as log-distributed.

    You can override this as discussed below.

Modifying output color mapping
------------------------------

You can override the default color set in multiple ways. First, you can specify the style
of a column to be `'category'`, `'linear'`, or `'log`' to override autodetection:

.. include:: ../_build_code/plot_well_metadata.py
    :code: python
    :start-line: 37
    :end-line: 38

.. figure:: ../_static/built_output/plot_metadata_custom/dox.svg
    :width: 60%
    :align: center

Importantly, you don't have to specify a style for each column plotted; it will default to style autodetection.

You can also specify an explicit colormap that specifies a color for each entry:

.. include:: ../_build_code/plot_well_metadata.py
    :code: python
    :start-line: 38
    :end-line: 43

.. figure:: ../_static/built_output/plot_metadata_custom/orientation.svg
    :width: 60%
    :align: center


Plotting non-96 wells
---------------------

By using the `plate_size` parameter, you can plot smaller plates, like a 6 well plate:

.. include:: ../_static/example_6well_plot_metadata.yaml
    :code: yaml

.. include:: ../_build_code/plot_well_metadata.py
    :code: python
    :start-line: 43
    :end-line: 44

.. figure:: ../_static/built_output/plot_metadata_custom/virus.svg
    :width: 60%
    :align: center

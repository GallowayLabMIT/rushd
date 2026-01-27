========================================
Generating table-like x-axis tick labels
========================================

When plotting experimental data, it is common to generate categorical plots organized by condition.
However, many times the full conditions are combinations of values for several different parameters.
This can result in very long condition names that are difficult to read, and rotating the labels does
not always help. Alternatively, conditions may be labeled with a short code (e.g., \"Condition 1\" or
\"A+B\"). This is less intuitive to understand, as it requires cross-referencing with a legend.

.. figure:: ../_static/built_output/generate_xticklabels/bad-formatting-examples.svg
    :width: 80%
    :align: center

A better solution would concisely communicate conditions based on its value of each relevant
parameter (or metadata). A table-like format for the axis label is well-suited to achieve this.

.. figure:: ../_static/built_output/generate_xticklabels/desired-output.svg
    :width: 40%
    :align: center

So, we'd like to convert default x-axis tick labels to custom ones, with each metadata value on a
separate line. We'd also like to do this automatically to ensure that the new labels line up
correctly with their corresponding condition, and to take advantage of any metadata already included
in our data.

We can use ``generate_xticklabels`` to do this:

.. autofunction:: rushd.plot.generate_xticklabels
    :noindex:

Consider an example with two cell lines and four conditions containing the presence or absence of
each of two small molecules:

.. include:: ../_build_code/generate_xticklabels.py
    :code: python
    :start-line: 42
    :end-line: 56

.. figure:: ../_static/built_output/generate_xticklabels/main-example.svg
    :width: 60%
    :align: center

You can also customize the text alignment using the ``align_annotation`` and ``align_ticklabels`` parameters:

.. include:: ../_build_code/generate_xticklabels.py
    :code: python
    :start-line: 61
    :end-line: 68

.. figure:: ../_static/built_output/generate_xticklabels/custom-alignment.svg
    :width: 60%
    :align: center

Notice that this integrates with Seaborn plots:

.. include:: ../_build_code/generate_xticklabels.py
    :code: python
    :start-line: 74
    :end-line: 84

.. figure:: ../_static/built_output/generate_xticklabels/seaborn-usage.svg
    :width: 60%
    :align: center

.. note::
    If creating multi-panel figures in Seaborn, either:

    - Set ``sharex=True`` and call ``generate_xticklabels`` on one set of axes
    - Set ``sharex=False`` and call ``generate_xticklabels`` for each set of axes

Scope toolbox for documentation purposes and comparisons
========================================================
Processing and comparison of time domain data similar to oscilloscopes in electronics. Typically used for technical comparisons in
 * Bachelor / Master / Ph.D. theses,
 * Scientific papers, 
 * Technical manuals, and
 * Measurement reports.

Overview
--------
Bring measurements from the oscilloscope and the circuit simulator into a standardized format. Edit the signals by shifting them in time (different zero points) or define the zero point for measuring equipment that can only record AC. Calculate the FFT or important values such as RMS, mean etc. Bring the originally different input formats into common plots to make comparisons easy.

.. image:: docs/source/figures/introduction.png

Getting started
---------------
Install this repository into your virtual environment (venv) or jupyter notebook:

::

    pip install pysignalscope

Use the toolbox in your python program:

::

    import pysignalscope as pss
    ...

Examples
--------
Have a look at the `example <examples/scope_example.py>`__, to see what you can do with this toolbox.

Naming convention
-------------------
This toolbox is divided into two modules: The functionality of an oscilloscope (``Scope``) and the functionality of an impedance analyzer (``Impedance``).

Scope
#####
The Scope module provides functionalities for editing and evaluating individual channels that are also provided by a real oscilloscope - just on a PC.
Scope creates, imports, edits or evaluates channels. The following prefixes apply:

- ``generate_``: Generates a new channel
- ``no prefix``: Is applied to a channel and results in a new channel (e.g. ``add()`` adds two channels)
- ``from_``: Generates a channel from an oscilloscope data set, a simulation program or a calculation (e.g. ``from_tektronix`` generates a channel from a tektronix scope file)
- ``calc_``: Calculates individual values from a channel (e.g. ``calc_rms()`` calculates the RMS from a given channel)
- ``plot_``: Plots channels in the desired arrangement (e.g. ``plot_channels()`` plots the given channels)

Impedance
#########




Documentation
---------------------------------------

Find the documentation `here <https://upb-lea.github.io/pySignalScope/intro.html>`__.


Bug Reports
-----------
Please use the issues report button within GitHub to report bugs.

Changelog
---------
Find the changelog `here <CHANGELOG.md>`__.

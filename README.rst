Scope toolbox for documentation purposes and comparisons
========================================================
pySignalScope processes and compares time domain data similar to oscilloscopes in electronics.
Signals can be filtered, derived, integrated or evaluated.
Furthermore, pySignalScope includes a module for evaluating and editing the impedance curves of an impedance analyzer or comparing them with the simulation.
The main purpose is the quick and easy evaluation of signals, as well as the targeted generation of images for technical documentation.
Some examples are:

- Bachelor / Master / Ph.D. theses,
- Scientific papers,
- Technical manuals, and
- Measurement reports.

Overview
--------
Bring measurements from the oscilloscope and the circuit simulator into a standardized ``Channel`` format.
Edit the signals by using the ``Scope functionality``: shifting them in time (different zero points) or define the zero point for measuring equipment that can only record AC.
Calculate the FFT or important values such as RMS, mean etc.
Bring the originally different input formats into common plots to make comparisons easy.

With the ``Impedance functionality``, ``ImpedanceChannels`` can be read in, edited and compared.
A conversion to e.g. the inductance value is possible with just one command.

.. image:: https://raw.githubusercontent.com/upb-lea/pySignalScope/main/docs/source/figures/introduction.png


Getting started
---------------
Install this repository into your virtual environment (venv) or jupyter notebook:

::

    pip install pysignalscope

Use the toolbox in your python program:

::

    import pysignalscope as pss
    ...

Example usage
-------------
pySignalScope helps to load, edit, display and analyze the signals.
The following application example loads a noisy measurement signal, which is first filtered.


::

    import pysignalscope as pss

    # Read curves from scope csv file
    [voltage_prim, voltage_sec, current_prim, current_sec] = pss.Scope.from_tektronix('scope_example_data_tek.csv')

    # Add labels and units to channel: This example considers the Channel 'current_prim' only
    current_prim = pss.Scope.modify(current_prim, label='current measured', unit='A', color='red')

    # Low pass filter the noisy current signal, modify the Channel attributes label, color and linestyle
    current_prim_filtered = pss.Scope.low_pass_filter(current_prim)
    current_prim_filtered = pss.Scope.modify(current_prim_filtered, label='current filtered', linestyle='--', color='green')

    # Make some modifications on the signal itself: data offset, time offset and factor to the data.
    # Short the channel to one period and add label, color and linestyle to the Channel
    current_prim_filtered_mod = pss.Scope.modify(current_prim_filtered, data_factor=1.3, data_offset=11, time_shift=2.5e-6)
    current_prim_filtered_mod = pss.Scope.short_to_period(current_prim_filtered_mod, f0=200000)
    current_prim_filtered_mod = pss.Scope.modify(current_prim_filtered_mod, label='current modified', linestyle='-', color='orange')

    # Plot channels, save as pdf
    fig1 = pss.Scope.plot_channels([current_prim, current_prim_filtered], [current_prim_filtered_mod], timebase='us')
    pss.save_figure(fig1, 'figure.pdf')

    # short channels to a single period, perform FFT for current waveforms
    current_prim = pss.Scope.short_to_period(current_prim, f0=200000)
    current_prim = pss.Scope.modify(current_prim, time_shift=5e-6)
    pss.Scope.fft(current_prim)

To simplify the display, colors, linestyle and the label can be attached to the object.
This is shown in the plot above.

The lower plot shows the post-processing of the filtered signal.
This is multiplied by a small gain, provided with an offset and shortened to a period duration.
The label, color and line style are changed.
The signals are then plotted with just one plot command.

.. image:: https://raw.githubusercontent.com/upb-lea/pySignalScope/main/docs/source/figures/function_overview.png

The functionality for the Impedance module is similar to the Scope module.
In here, ``ImpedanceChannel`` objects can be loaded from different sources, which can be a ``.csv`` measurement file from an impedance analyzer or a computer generated curve.
``ImpedanceChannel`` objects can be modified in attributes and data, plotted and equivalent circuit parameters can be obtained from measurements.

.. image:: https://raw.githubusercontent.com/upb-lea/pySignalScope/main/docs/source/figures/impedance_function_overview.png

Have a look at the `Scope example <https://github.com/upb-lea/pySignalScope/blob/main/examples/scope_example.py>`__ and at the `Impedance example <https://github.com/upb-lea/pySignalScope/blob/main/examples/impedance_example.py>`__ to see what you can do with this toolbox.

Naming convention
-------------------
This toolbox is divided into two modules: The functionality of an oscilloscope (``Scope``) and the functionality of an impedance analyzer (``Impedance``).

Scope
#####
The ``Scope`` module provides functionalities for editing and evaluating individual channels that are also provided by a real oscilloscope - just on a PC.
``Scope`` creates, imports, edits or evaluates ``Channels``. The following prefixes apply:

- ``generate_``: Generates a new ``Channel``
- ``no prefix``: Is applied to a ``Channel`` and results in a new ``Channel`` (e.g. ``add()`` adds two channels)
- ``from_``: Generates a ``Channel`` from an oscilloscope data set, a simulation program or a calculation (e.g. ``from_tektronix`` generates a ``Channel`` from a tektronix scope file)
- ``calc_``: Calculates individual values from a ``Channel`` (e.g. ``calc_rms()`` calculates the RMS from a given ``Channel``)
- ``plot_``: Plots channels in the desired arrangement (e.g. ``plot_channels()`` plots the given ``Channels``)

Impedance
#########
The ``Impedance`` module provides functionalities to evaluate impedance curves.
``Impedance`` creates, imports, edits or evaluates ``ImpedanceChannel``.

- ``generate_``: Generates a new ``ImpedanceChannel``
- ``no prefix``: Is applied to a ``ImpedanceChannel`` and results in a new ``ImpedanceChannel`` (e.g. ``modify()`` modifies an ``ImpedanceChannel``)
- ``from_``: Generates a ``ImpedanceChannel`` from an impedance analyzer data set, a simulation program or a calculation (e.g. ``from_waynekerr`` generates a ``ImpedanceChannel`` from a real measurement file)
- ``calc_``: Calculates individual values from a ``ImpedanceChannel`` (e.g. ``calc_rlc()`` calculates the equivalent resistance, inductance and capacitance)
- ``plot_``: Plots ``ImpedanceChannel`` (e.g. ``plot_impedance()`` plots the given ``ImpedanceChannels``)



Documentation
---------------------------------------

Find the documentation `here <https://upb-lea.github.io/pySignalScope/intro.html>`__.


Bug Reports
-----------
Please use the issues report button within GitHub to report bugs.

Changelog
---------
Find the changelog `here <CHANGELOG.md>`__.

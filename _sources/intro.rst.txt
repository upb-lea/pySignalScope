.. sectnum::

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

.. image:: figures/introduction.png

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
pySignalScope helps to load, edit, display and analyze the signals. The following application example loads a noisy measurement signal, which is first filtered. To simplify the display, colors, linestyle and the label can be attached to the object. This is shown in the plot above.


The lower plot shows the post-processing of the filtered signal. This is multiplied by a small gain, provided with an offset and shortened to a period duration. The label, color and line style are changed. The signals are then plotted with just one plot command.


.. image:: figures/function_overview.png



Examples
--------
Have a look at the `example <https://github.com/upb-lea/pySignalScope/blob/main/examples/scope_example.py>`__, to see what you can do with this toolbox.

Bug Reports
-----------
Please use the issues report button within GitHub to report bugs.

Changelog
---------
Find the changelog `here <https://github.com/upb-lea/pySignalScope/blob/main/CHANGELOG.md>`__.



pySignalScope function documentation
==================================================
.. currentmodule:: pysignalscope.scope

.. autoclass:: pysignalscope.Scope
   :members: 
   
.. autoclass:: pysignalscope.HandleScope
   :members: 
     
.. autoclass:: pysignalscope.Impedance
   :members:
   
.. autoclass:: pysignalscope.HandleImpedance
   :members:
  
.. automodule:: pysignalscope.functions
   :members:

.. automodule:: pysignalscope.generalplotsettings
   :members:

.. automodule:: pysignalscope.colors
   :members:

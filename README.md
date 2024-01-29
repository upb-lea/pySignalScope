# Scope toolbox for documentation purposes and comparisons
Processing and comparison of time domain data similar to oscilloscopes in electronics. Typically used for technical comparisons in
 * Bachelor / Master / Ph.D. theses,
 * Scientific papers, 
 * Technical manuals, and
 * Measurement reports.

# Overview
Bring measurements from the oscilloscope and the circuit simulator into a standardized format. Edit the signals by shifting them in time (different zero points) or define the zero point for measuring equipment that can only record AC. Calculate the FFT or important values such as RMS, mean etc. Bring the originally different input formats into common plots to make comparisons easy.
![](docs/source/figures/introduction.png)

# Getting started
It is currently only possible to obtain the current development version:
```
cd /Documents/Folder/of/Interest   
git clone git@github.com:upb-lea/pySignalScope.git
```
Install this repository into your virtual environment (venv) or jupyter notebook:
```
pip install -e .
```
Use the toolbox in your python program:
```
import pysignalscope as pss

...
```

# Examples


# Bug Reports
Please use the issues report button within GitHub to report bugs.

# Changelog
Find the changelog [here](CHANGELOG.md).

"""Set general plot settings, like LaTeX font."""
# 3rd party libraries
from matplotlib import pyplot as plt

# own libraries
from pysignalscope.impedance import Impedance
from pysignalscope.scope import Channel
def global_plot_settings_unit_delimiter_slash() -> None:
    """
    Set the plot labeling delimiter to "/".

    e.g. Voltage / V
    """
    Impedance.unit_separator_plot = "/"
    Channel.unit_separator_plot = "/"

def global_plot_settings_unit_delimiter_in() -> None:
    """
    Set the plot labeling delimiter to "in".

    e.g. Voltage in V
    """
    Impedance.unit_separator_plot = "in"
    Channel.unit_separator_plot = "in"


def global_plot_settings_font_latex():
    """Set the plot fonts to LaTeX-font."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })


def global_plot_settings_font_sansserif():
    """Set the plot fonts to Sans-Serif-Font."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

def update_font_size(font_size: int = 11):
    """Update the figure font size.

    :param font_size: font size
    :type font_size: int
    """
    plt.rcParams.update({'font.size': font_size})

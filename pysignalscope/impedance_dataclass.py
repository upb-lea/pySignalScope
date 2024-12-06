"""Definition of the impedance dataclass."""

# python libraries
import dataclasses
from typing import Union

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class Impedance:
    """Dataclass to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    # mandatory data
    channel_frequency: np.array  #: mandatory: frequency data (mandatory)
    channel_impedance: np.array  #: mandatory: impedance data (mandatory)
    channel_phase: np.array  #: mandatory: phase data (mandatory)

    # optional data
    channel_label: Union[str]  #: channel label displayed in a plot (optional)
    channel_unit: Union[str]  #: channel unit displayed in a plot (optional)
    channel_color: Union[str]  #: channel color displayed in a plot (optional)
    channel_linestyle: Union[str]  #: channel linestyle displayed in a plot (optional)

    # meta data
    channel_source: Union[str]  #: channel source, additional meta data (optional)

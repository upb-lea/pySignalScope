"""Definition of the impedance dataclass."""

# python libraries
import dataclasses
from typing import Union

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class Impedance:
    """Dataclass to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    channel_frequency: np.array
    channel_impedance: np.array
    channel_phase: np.array
    channel_label: Union[str]
    channel_unit: Union[str]
    channel_color: Union[str]
    channel_source: Union[str]
    channel_linestyle: Union[str]

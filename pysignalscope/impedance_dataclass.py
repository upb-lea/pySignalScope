"""Definition of the impedance dataclass."""

# python libraries
import dataclasses
from typing import Optional

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class ImpedanceCurve:
    """Dataclass to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    channel_frequency: np.array
    channel_impedance: np.array
    channel_phase: np.array
    channel_label: Optional[str]
    channel_unit: Optional[str]
    channel_color: Optional[str]
    channel_source: Optional[str]
    channel_linestyle: Optional[str]

"""Definition of the impedance dataclass."""

# python libraries
import dataclasses
from typing import Union

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(eq=False)
class ImpedanceCurve:
    """Dataclass for impedance objects in a special format, to keep labels, units and voltages belonging to a certain curve."""

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

    def __eq__(self, other):
        """Compare two Channel objects."""
        def array_eq(arr1, arr2):
            return (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray) and \
                    arr1.shape == arr2.shape and (arr1 == arr2).all())

        if not isinstance(other, ImpedanceCurve):
            return NotImplemented("Type Channel must be compared to type Channel.")
        return (array_eq(self.channel_frequency, other.channel_frequency) and \
                array_eq(self.channel_impedance, other.channel_impedance) and \
                array_eq(self.channel_phase, other.channel_phase) and \
                (self.channel_label == other.channel_label) and \
                (self.channel_unit == other.channel_unit) and \
                (self.channel_color == other.channel_color) and \
                (self.channel_source == other.channel_source) and \
                (self.channel_linestyle == other.channel_linestyle))

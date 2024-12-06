"""Definition of the impedance dataclass."""

# python libraries
import dataclasses
from typing import Optional

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(eq=False)
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

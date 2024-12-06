"""Definition of the impedance dataclass."""

# python libraries
import dataclasses
from typing import Union

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(eq=False)
class ImpedanceChannel:
    """Dataclass for impedance objects in a special format, to keep labels, units and voltages belonging to a certain curve."""

    # mandatory data
    frequency: np.array  #: mandatory: frequency data (mandatory)
    impedance: np.array  #: mandatory: impedance data (mandatory)
    phase: np.array  #: mandatory: phase data (mandatory)

    # optional data
    label: Union[str]  #: channel label displayed in a plot (optional)
    unit: Union[str]  #: channel unit displayed in a plot (optional)
    color: Union[str]  #: channel color displayed in a plot (optional)
    linestyle: Union[str]  #: channel linestyle displayed in a plot (optional)

    # meta data
    source: Union[str]  #: channel source, additional meta data (optional)

    def __eq__(self, other):
        """Compare two Channel objects."""
        def array_eq(arr1, arr2):
            return (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray) and \
                    arr1.shape == arr2.shape and (arr1 == arr2).all())

        if not isinstance(other, ImpedanceChannel):
            return NotImplemented("Type Channel must be compared to type Channel.")
        return (array_eq(self.frequency, other.frequency) and \
                array_eq(self.impedance, other.impedance) and \
                array_eq(self.phase, other.phase) and \
                (self.label == other.label) and \
                (self.unit == other.unit) and \
                (self.color == other.color) and \
                (self.source == other.source) and \
                (self.linestyle == other.linestyle))

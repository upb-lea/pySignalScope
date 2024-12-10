"""Definition of the impedance dataclass."""

# python libraries
import dataclasses
from typing import Union, Optional

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(eq=False)
class ImpedanceChannel:
    """Dataclass for ImpedanceChannel objects in a special format, to keep labels, units and voltages belonging to a certain curve."""

    # mandatory data
    frequency: np.array  #: mandatory: frequency data (mandatory)
    impedance: np.array  #: mandatory: impedance data (mandatory)
    phase_deg: np.array  #: mandatory: phase data in degree (mandatory)

    # optional data
    label: Optional[str]  #: channel label displayed in a plot (optional)
    unit: Optional[str]  #: channel unit displayed in a plot (optional)
    color: Union[str, tuple, None]  #: channel color displayed in a plot (optional)
    linestyle: Optional[str]  #: channel linestyle displayed in a plot (optional)

    # meta data
    source: Optional[str]  #: channel source, additional meta data (optional)

    def __eq__(self, other):
        """Compare two Channel objects."""
        def array_eq(arr1, arr2):
            return (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray) and \
                    arr1.shape == arr2.shape and (arr1 == arr2).all())

        if not isinstance(other, ImpedanceChannel):
            return NotImplemented("Type Channel must be compared to type Channel.")
        return (array_eq(self.frequency, other.frequency) and \
                array_eq(self.impedance, other.impedance) and \
                array_eq(self.phase_deg, other.phase_deg) and \
                (self.label == other.label) and \
                (self.unit == other.unit) and \
                (self.color == other.color) and \
                (self.source == other.source) and \
                (self.linestyle == other.linestyle))

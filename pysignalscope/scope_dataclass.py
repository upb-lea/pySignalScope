"""Definition of the channel dataclass."""

# python libraries
import dataclasses
from typing import Optional

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(eq=False)
class Channel:
    """Dataclass for Channel objects in a special format, to keep labels, units and voltages belonging to a certain curve."""

    # mandatory measurement data
    time: np.array  #: time series of the channel (mandatory)
    data: np.array  #: data series of the channel (mandatory)

    # optional data
    label: Optional[str]  #: channel label displayed in a plot (optional)
    unit: Optional[str]  #: channel unit displayed in a plot (optional)
    color: Optional[str]  #: channel color in a plot (optional)
    linestyle: Optional[str]  #: channel linestyle in a plot (optional)

    # meta data
    source: Optional[str]  #: channel source, additional meta data (optional)
    modulename: Optional[str]

    def __eq__(self, other):
        """Compare two Channel objects."""
        def array_eq(arr1, arr2):
            return (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray) and \
                    arr1.shape == arr2.shape and (arr1 == arr2).all())

        if not isinstance(other, Channel):
            return NotImplemented("Type Channel must be compared to type Channel.")
        return (array_eq(self.time, other.time) and \
                array_eq(self.data, other.data) and \
                (self.label == other.label) and \
                (self.unit == other.unit) and \
                (self.color == other.color) and \
                (self.source == other.source) and \
                (self.linestyle == other.linestyle))

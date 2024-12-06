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
    channel_time: np.array  #: time series of the channel (mandatory)
    channel_data: np.array  #: data series of the channel (mandatory)

    # optional data
    channel_label: Optional[str]  #: channel label displayed in a plot (optional)
    channel_unit: Optional[str]  #: channel unit displayed in a plot (optional)
    channel_color: Optional[str]  #: channel color in a plot (optional)
    channel_linestyle: Optional[str]  #: channel linestyle in a plot (optional)

    # meta data
    channel_source: Optional[str]  #: channel source, additional meta data (optional)
    modulename: Optional[str]

    def __eq__(self, other):
        """Compare two Channel objects."""
        def array_eq(arr1, arr2):
            return (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray) and \
                    arr1.shape == arr2.shape and (arr1 == arr2).all())

        if not isinstance(other, Channel):
            return NotImplemented("Type Channel must be compared to type Channel.")
        return (array_eq(self.channel_time, other.channel_time) and \
                array_eq(self.channel_data, other.channel_data) and \
                (self.channel_label == other.channel_label) and \
                (self.channel_unit == other.channel_unit) and \
                (self.channel_color == other.channel_color) and \
                (self.channel_source == other.channel_source) and \
                (self.channel_linestyle == other.channel_linestyle))

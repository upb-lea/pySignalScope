"""Definition of the scope dataclass."""

# python libraries
import dataclasses
from typing import Optional

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class Scope:
    """Scope data class. Attributes as the following."""

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

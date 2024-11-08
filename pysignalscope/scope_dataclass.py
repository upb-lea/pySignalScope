"""Definition of the scope dataclass."""

# python libraries
import dataclasses
from typing import Optional

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class Scope:
    """Scope data class."""

    # mandatory measurement data
    channel_time: np.array
    channel_data: np.array

    # optional data
    channel_label: Optional[str]
    channel_unit: Optional[str]
    channel_color: Optional[str]
    channel_source: Optional[str]
    channel_linestyle: Optional[str]

    # meta data
    modulename: Optional[str]

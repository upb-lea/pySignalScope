"""Classes and methods to process scope data (from real scopes or from simulation tools) like in a real scope."""
# python libraries
import copy
import os.path
import logging
from typing import Union, List, Tuple, Optional, Any
import pickle

# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt
from lecroyutils.control import LecroyScope
from scipy import signal
from findiff import FinDiff

# own libraries
import pysignalscope.functions as functions
from pysignalscope.logconfig import setup_logging
from pysignalscope.scope_dataclass import Channel
from pysignalscope.channelshift import ScopeChShift as scope_ch_shift

# - Logging setup ---------------------------------------------------------------------------------
setup_logging()

# Modul name für static methods
class_modulename = "scope"

# - Class definition ------------------------------------------------------------------------------

class Scope:
    """Class to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    @staticmethod
    def generate_channel(channel_time: Union[List[float], np.ndarray], channel_data: Union[List[float], np.ndarray],
                         channel_label: Optional[str] = None, channel_unit: Optional[str] = None, channel_color: Union[str, tuple, None] = None,
                         channel_source: Optional[str] = None, channel_linestyle: Optional[str] = None) -> Channel:
        """
        Generate a scope object.

        :param channel_time: time series
        :type channel_time: Union[List[float], np.ndarray]
        :param channel_data: channel data
        :type channel_data: Union[List[float], np.ndarray]
        :param channel_label: channel label
        :type channel_label: Optional[str]
        :param channel_unit: channel unit
        :type channel_unit: Optional[str]
        :param channel_color: channel color
        :type channel_color: Optional[str]
        :param channel_source: channel source
        :type channel_source: Optional[str]
        :param channel_linestyle: channel linestyle
        :type channel_linestyle: Optional[str]
        """
        # check channel_time for a valid type, convert to numpy if necessary
        if isinstance(channel_time, List):
            channel_time = np.array(channel_time)
        elif isinstance(channel_time, np.ndarray):
            channel_time = channel_time
        else:
            raise TypeError("channel_time must be type list or ArrayLike.")
        # check channel_data for a valid type, convert to numpy if necessary
        if isinstance(channel_data, List):
            channel_data = np.array(channel_data)
        elif isinstance(channel_data, np.ndarray):
            channel_data = channel_data
        else:
            raise TypeError("channel_data must be type list or ArrayLike")
        # check for single non-allowed values in channel_data
        if np.any(np.isnan(channel_data)):
            raise ValueError("NaN is not allowed in channel_data.")
        if np.any(np.isinf(channel_data)):
            raise ValueError("inf is not allowed in channel_data.")
        # check for empty data
        if channel_time.size == 0:
            raise ValueError("Not allowed: channel_time is empty")
        if channel_data.size == 0:
            raise ValueError("Not allowed: channel_data is empty")
        # check if channel_time and channel_data have the same length
        if len(channel_time) != len(channel_data):
            raise ValueError("channel_time and channel_data must be same length.")
        # check if channel_time is strictly increasing
        if not np.all(np.diff(channel_time) > 0):
            raise ValueError("channel time not strictly increasing.")

        # check channel_label for a valid type
        if isinstance(channel_label, str) or isinstance(channel_color, tuple) or channel_label is None:
            channel_label = channel_label
        else:
            raise TypeError("channel_label must be type str or None.")
        # check channel unit for a valid type
        if isinstance(channel_unit, str) or channel_unit is None:
            channel_unit = channel_unit
        else:
            raise TypeError("channel_unit must be type str or None.")
        # check channel_color for a valid type
        if isinstance(channel_color, str) or channel_color is None or isinstance(channel_color, tuple):
            channel_color = channel_color
        else:
            raise TypeError("channel_color must be type str or None.")
        # check channel_source for a valid type
        if isinstance(channel_source, str) or channel_source is None:
            channel_source = channel_source
        else:
            raise TypeError("channel_source must be type str or None.")
        # check channel_linestyle for a valid type
        if isinstance(channel_linestyle, str) or channel_linestyle is None:
            channel_linestyle = channel_linestyle
        else:
            raise TypeError("channel_linestyle must be type str or None.")

        return Channel(time=channel_time,
                       data=channel_data,
                       label=channel_label,
                       unit=channel_unit,
                       color=channel_color,
                       source=channel_source,
                       linestyle=channel_linestyle,
                       modulename=class_modulename)

    # - Method modify ------------------------------------------------------------------------------

    @staticmethod
    def modify(channel: Channel, channel_data_factor: Optional[float] = None, channel_data_offset: Optional[float] = None,
               channel_label: Optional[str] = None, channel_unit: Optional[str] = None, channel_color: Union[str, tuple, None] = None,
               channel_source: Optional[str] = None, channel_time_shift: Optional[float] = None,
               channel_time_shift_rotate: Optional[float] = None,
               channel_time_cut_min: Optional[float] = None, channel_time_cut_max: Optional[float] = None,
               channel_linestyle: Optional[str] = None) -> Channel:
        """
        Modify channel data like metadata or add a factor or offset to channel data.

        Useful for classes with channel_time/data, but without labels or units.

        :param channel: Scope channel object
        :type channel: Channel
        :param channel_data_factor: multiply self.channel_data by channel_data_factor
        :type channel_data_factor: float
        :param channel_data_offset: add an offset to self.channel_data
        :type channel_data_offset: float
        :param channel_label: label to add to the Channel-class
        :type channel_label: str
        :param channel_unit: unit to add to the Channel-class
        :type channel_unit: str
        :param channel_color: Color of a channel
        :type channel_color: str
        :param channel_source: Source of a channel, e.g. 'GeckoCIRCUITS', 'Numpy', 'Tektronix-Scope', ...
        :type channel_source: str
        :param channel_time_shift: add time to the time base
        :type channel_time_shift: float
        :param channel_time_shift_rotate: shifts a signal by the given time, but the end of the signal will
            come to the beginning of the signal. Only recommended for periodic signals!
        :type channel_time_shift_rotate: float
        :param channel_time_cut_min: removes all time units smaller than the given one
        :type channel_time_cut_min: float
        :param channel_time_cut_max: removes all time units bigger than the given one
        :type channel_time_cut_max: float
        :param channel_linestyle: channel linestyle, e.g. '--'
        :type channel_linestyle: str
        :return: Scope object
        :rtype: Channel
        """
        channel_modified = copy.deepcopy(channel)

        if isinstance(channel_label, str):
            channel_modified.label = channel_label
            modify_flag = True
        elif channel_label is None:
            pass
        else:
            raise TypeError("channel_label must be type str or None")
        if isinstance(channel_unit, str):
            channel_modified.unit = channel_unit
            modify_flag = True
        elif channel_unit is None:
            pass
        else:
            raise TypeError("channel_unit must be type str or None")
        if isinstance(channel_data_factor, (int, float)):
            channel_modified.data = channel_modified.data * channel_data_factor
            modify_flag = True
        elif channel_data_factor is None:
            pass
        else:
            raise TypeError("channel_data_factor must be type float or None")
        if isinstance(channel_data_offset, (int, float)):
            channel_modified.data = channel_modified.data + channel_data_offset
            modify_flag = True
        elif channel_data_offset is None:
            pass
        else:
            raise TypeError("channel_data_offset must be type float or None")
        if isinstance(channel_color, str) or isinstance(channel_color, tuple):
            channel_modified.color = channel_color
            modify_flag = True
        elif channel_color is None:
            pass
        else:
            raise TypeError("channel_color must be type str or tuple or None")
        if isinstance(channel_source, str):
            channel_modified.source = channel_source
            modify_flag = True
        elif channel_source is None:
            pass
        else:
            raise TypeError("channel_source must be type str or None")
        if isinstance(channel_time_shift, (int, float)):
            channel_modified.time = channel_modified.time + channel_time_shift
            modify_flag = True
        elif channel_time_shift is None:
            pass
        else:
            raise TypeError("channel_time_shift must be type float or None")
        if isinstance(channel_time_shift_rotate, (int, float)):
            # figure out current max time
            current_max_time = channel_modified.time[-1]
            current_period = current_max_time - channel_modified.time[0]
            # shift all times
            channel_modified.time = channel_modified.time + channel_time_shift_rotate
            channel_modified.time[channel_modified.time > current_max_time] = (
                channel_modified.time[channel_modified.time > current_max_time] - current_period)
            # due to rolling time-shift, channel_time and channel_data needs to be re-sorted.
            new_index = np.argsort(channel_modified.time)
            channel_modified.time = np.array(channel_modified.time)[new_index]
            channel_modified.data = np.array(channel_modified.data)[new_index]
            modify_flag = True
        elif channel_time_shift_rotate is None:
            pass
        else:
            raise TypeError("channel_time_shift_rotate must be type str or None")

        if isinstance(channel_time_cut_min, (int, float)):
            index_list_to_remove = []
            if channel_time_cut_min < channel_modified.time[0]:
                raise ValueError(f"channel_cut_time_min ({channel_time_cut_min}) < start of channel_time ({channel_modified.time[0]}). "
                                 f"This is not allowed!")
            for count, value in enumerate(channel_modified.time):
                if value < channel_time_cut_min:
                    index_list_to_remove.append(count)
            channel_modified.time = np.delete(channel_modified.time, index_list_to_remove)
            channel_modified.data = np.delete(channel_modified.data, index_list_to_remove)
            modify_flag = True
        elif channel_time_cut_min is None:
            pass
        else:
            raise TypeError("channel_time_cut_min must be type float or None")

        if isinstance(channel_time_cut_max, (int, float)):
            index_list_to_remove = []
            if channel_time_cut_max > channel_modified.time[-1]:
                raise ValueError(f"channel_cut_time_max ({channel_time_cut_max}) > end of channel_time ({channel_modified.time[-1]}). "
                                 f"This is not allowed!")
            for count, value in enumerate(channel_modified.time):
                if value > channel_time_cut_max:
                    index_list_to_remove.append(count)
            channel_modified.time = np.delete(channel_modified.time, index_list_to_remove)
            channel_modified.data = np.delete(channel_modified.data, index_list_to_remove)
            modify_flag = True
        elif channel_time_cut_max is None:
            pass
        else:
            raise TypeError("channel_time_cut_max must be type float or None")
        if isinstance(channel_linestyle, str):
            channel_modified.linestyle = channel_linestyle
            modify_flag = True
        elif channel_linestyle is None:
            pass
        else:
            raise TypeError("channel_linestyle must be type str or None")

        # Log flow control
        logging.debug(f"{channel_modified.modulename} :FlCtl")
        # Log, if no modification is requested
        if not modify_flag:
            logging.info(f"{channel_modified.modulename} : No modification is requested", )

        return channel_modified

    @staticmethod
    def copy(channel: Channel) -> Channel:
        """Create a deepcopy of Channel.

        :param channel: Scope channel object
        :type channel: Channel
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be type Scope.")
        return copy.deepcopy(channel)

    @staticmethod
    def from_tektronix(csv_file: str) -> List['Channel']:
        """
        Translate tektronix csv-file to a tuple of Channel.

        Note: Returns a tuple with four Channels (Tektronix stores multiple channel data in single .csv-file,
        this results to return of a tuple containing Channel's)

        :param csv_file: csv-file from tektronix scope
        :type csv_file: str
        :return: tuple of Channel, depending on the channel count stored in the .csv-file
        :rtype: list[Channel, Scope, Scope, Scope]

        :Example:

        >>> import pysignalscope as pss
        >>> [voltage, current_prim, current_sec] = pss.Scope.from_tektronix('/path/to/tektronix/file/tek0000.csv')
        """
        channel_source = 'Tektronix scope'

        if not isinstance(csv_file, str):
            raise TypeError("csv_file must be type str to show the full filepath.")

        file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=15)
        channel_counts = file.shape[1] - 1
        time = file[:, 0]

        channel_list = []
        for channel_count in range(1, channel_counts + 1):
            channel_list.append(Scope.generate_channel(
                channel_time=time, channel_data=file[:, channel_count], channel_source=channel_source,
                channel_label=None, channel_unit=None, channel_color=None, channel_linestyle=None))

        # Log user error Empty csv-file
        if channel_count == 0:
            logging.info(f"{class_modulename} : Invalid file or file without content")

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Channel counts {channel_counts}, File {csv_file}")

        return channel_list

    @staticmethod
    def from_tektronix_mso58(*csv_files: str) -> List['Channel']:
        """
        Translate tektronix csv-files to a list of Channel class objects.

        Note: insert multiple .csv-files to get a list of all channels.

        :param csv_files: csv-file from tektronix scope
        :type csv_files: str
        :return: List of Scope objects
        :rtype: List['Scope']

        :Example single channel:

        >>> import pysignalscope as pss
        >>> [current_prim] = pss.Scope.from_tektronix_mso58('/path/to/lecroy/files/current_prim.csv')

        :Example multiple channels channel:

        >>> import pysignalscope as pss
        >>> [current_prim, current_sec] = pss.Scope.from_tektronix_mso58('/path/one/current_prim.csv', '/path/two/current_sec.csv')
        """
        channel_source = 'Tektronix scope MSO58'

        tektronix_channels = []
        for csv_file in csv_files:
            if not isinstance(csv_file, str):
                raise TypeError("csv_file must be type str to show the full filepath.")
            file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=24)
            time = file[:, 0]
            ch1_data = file[:, 1]

            # Log user error Empty csv-file
            if not ch1_data:
                logging.info(f"{class_modulename} : file {csv_file}->Invalid file or file without content")

            tektronix_channels.append(Scope.generate_channel(
                time, ch1_data, channel_source=channel_source, channel_label=os.path.basename(csv_file).replace('.csv', ''),
                channel_unit=None, channel_linestyle=None, channel_color=None))

        # Log user error Empty csv-file
        if not tektronix_channels:
            logging.info(f"{class_modulename} : file {csv_files}-> Invalid file list or files without content")

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl {csv_files}")

        return tektronix_channels

    @staticmethod
    def from_tektronix_mso58_multichannel(csv_file: str) -> List['Channel']:
        """
        Translate tektronix csv-files to a list of Channel class objects.

        Note: insert multiple .csv-files to get a list of all channels.

        :param csv_file: csv-file from tektronix scope
        :type csv_file: str
        :return: List of Scope objects
        :rtype: List['Scope']

        :Example multiple channel csv-file:

        >>> import pysignalscope as pss
        >>> [current_prim, current_sec] = pss.Scope.from_tektronix_mso58_multichannel('/path/to/lecroy/files/currents.csv')
        """
        channel_source = 'Tektronix scope MSO58'

        if not isinstance(csv_file, str):
            raise TypeError("csv_file must be type str to show the full filepath.")

        file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=24)
        channel_counts = file.shape[1] - 1
        time = file[:, 0]

        channel_list = []
        for channel_count in range(1, channel_counts + 1):
            channel_list.append(Scope.generate_channel(
                time, file[:, channel_count], channel_source=channel_source, channel_color=None, channel_linestyle=None,
                channel_unit=None, channel_label=None))

        # Log user error Empty csv-file
        if channel_count == 0:
            logging.info(f"{class_modulename} : Invalid file or file without content", class_modulename)

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Channel counts {channel_count},File {csv_file}")

        return channel_list

    @staticmethod
    def from_lecroy(*csv_files: str) -> List['Channel']:
        """
        Translate LeCroy csv-files to a list of Channel class objects.

        Note: insert multiple .csv-files to get a list of all channels

        :param csv_files: csv-file from tektronix scope
        :type csv_files: str
        :return: List of scope objects
        :rtype: List['Scope']

        :Example single channel:

        >>> import pysignalscope as pss
        >>> [current_prim] = pss.Scope.from_lecroy('/path/to/lecroy/files/current_prim.csv')

        :Example multiple channels channel:

        >>> import pysignalscope as pss
        >>> [current_prim, current_sec] = pss.Scope.from_lecroy('/path/one/current_prim.csv', '/path/two/current_sec.csv')
        """
        channel_source = 'LeCroy scope'

        lecroy_channel = []
        for csv_file in csv_files:
            if not isinstance(csv_file, str):
                raise TypeError("csv_file must be type str to show the full filepath.")
            file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=5)
            time = file[:, 0]
            ch1_data = file[:, 1]

            lecroy_channel.append(Scope.generate_channel(
                channel_time=time, channel_data=ch1_data, channel_source=channel_source,
                channel_label=os.path.basename(csv_file).replace('.csv', ''), channel_unit=None,
                channel_color=None, channel_linestyle=None))

        # Log user error Empty csv-file
        if not lecroy_channel:
            logging.info(f"{class_modulename} : {csv_files} Invalid file list or files without content")

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl {csv_file}")

        return lecroy_channel

    @staticmethod
    def from_lecroy_remote(channel_number: int, ip_address: str, channel_label: str):
        """
        Get the data of a LeCroy oscilloscope and return a scope object with the collected data.

        :param channel_number: number of the channel
        :type channel_number: int
        :param ip_address: ip-address of the oscilloscope
        :type ip_address: str
        :param channel_label: label name of channel
        :type channel_label: str
        :return: Scope object with collected data
        :rtype: 'Scope'
        """
        if not isinstance(channel_number, int):
            raise TypeError("channel_number must be type int.")
        if not isinstance(ip_address, str):
            raise TypeError("ip_address must be type str.")
        if not isinstance(channel_label, str):
            raise TypeError("channel_label must be type str.")

        channel_source = "LeCroy scope"

        scope = LecroyScope(ip_address)

        if channel_number == 1:
            channel = "C1"
        elif channel_number == 2:
            channel = "C2"
        elif channel_number == 3:
            channel = "C3"
        elif channel_number == 4:
            channel = "C4"
        elif channel_number == 5:
            channel = "C5"
        elif channel_number == 6:
            channel = "C6"
        elif channel_number == 7:
            channel = "C7"
        elif channel_number == 8:
            channel = "C8"
        else:  # "else-case"
            channel = None
            # Log user info
            logging.info(f"{class_modulename} : Requested channel number {channel_number} is out of range (1-8)")
            print("No fitting channel found!")

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Channel number: {channel_number} Assigned  channel: {channel}")

        if channel is not None:
            data = scope.waveform(channel)
            return Scope.generate_channel(
                channel_time=data.x, channel_data=data.y, channel_source=channel_source, channel_label=channel_label,
                channel_color=None, channel_linestyle=None, channel_unit=None)

    @staticmethod
    def from_numpy(period_vector_t_i: np.ndarray, mode: str = 'rad', f0: Union[float, None] = None,
                   channel_label: Optional[str] = None, channel_unit: Optional[str] = None) -> 'Channel':
        """
        Bring a numpy or list array to an instance of Channel.

        :param period_vector_t_i: input vector np.array([time], [signal])
        :type period_vector_t_i: npt.ArrayLike
        :param mode: 'rad' [default], 'deg' or 'time'
        :type mode: str
        :param f0: fundamental frequency in Hz
        :type f0: float
        :param channel_label: channel label
        :type channel_label: str
        :param channel_unit: channel unit
        :type channel_unit: str

        :Example:

        >>> import pysignalscope as pss
        >>> import numpy as np
        >>> channel = pss.Scope.from_numpy(np.array([[0, 5e-3, 10e-3, 15e-3, 20e-3], [1, -1, 1, -1, 1]]), f0=100000, mode='time')
        """
        # changes period_vector_t_i to a float array. e.g. the user inserts a vector like this
        # [[0, 90, 180, 270, 360], [1, -1, 1, -1, 1]], with degree-mode, there are only integers inside.
        # when re-assigning the vector by switching from degree to time, the floats will be changed to an integers,
        # what is for most high frequency signals a time-vector consisting with only zeros.
        period_vector_t_i = period_vector_t_i.astype(float)

        # check for correct input parameter
        if (mode == 'rad' or mode == 'deg') and f0 is None:
            raise ValueError("if mode is 'rad' or 'deg', a fundamental frequency f0 must be set")

        # mode pre-calculation
        if mode == 'rad' and f0 is not None:
            period_vector_t_i[0] = period_vector_t_i[0] / (2 * np.pi * f0)
        elif mode == 'deg' and f0 is not None:
            period_vector_t_i[0] = period_vector_t_i[0] / (360 * f0)
        elif mode != 'time':
            raise ValueError("Mode not available. Choose: 'rad', 'deg', 'time'")

        single_dataset_channel = Scope.generate_channel(period_vector_t_i[0], period_vector_t_i[1],
                                                        channel_label=channel_label, channel_unit=channel_unit, channel_color=None,
                                                        channel_linestyle=None, channel_source=None)

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of Data: 1")

        return single_dataset_channel

    @staticmethod
    def from_geckocircuits(txt_datafile: str, f0: Optional[float] = None) -> List['Channel']:
        """
        Convert a gecko simulation file to Channel.

        :param txt_datafile: path to text file, generated by geckoCIRCUITS
        :type txt_datafile: str
        :param f0: fundamental frequency [optional]
        :type f0: float
        :return: List of Channels
        :rtype: list[Channel]
        """
        if not isinstance(txt_datafile, str):
            raise TypeError("txt_datafile must be type str to show the full filepath.")
        if not isinstance(f0, (float, int)) != f0 is not None:
            raise TypeError("f0 must be type float/int/None.")

        channel_source = 'GeckoCIRCUITS simulation'

        # Read variables from first line in gecko output file
        file = open(txt_datafile, 'r')
        header_line_variables = file.readline(-1)
        variables = header_line_variables.replace('# ', '').split(' ')
        variables.pop()

        # Read simulation data from gecko output file, cut the data and transfer it to Channel
        txt_data = np.genfromtxt(txt_datafile, delimiter=' ')

        timestep = round(txt_data[:, 0][1] - txt_data[:, 0][0], 12)
        if f0 is not None:
            points_last_period = int(1 / f0 / timestep) + 1
            channel_data_last_period = txt_data[-points_last_period:]
            time_modified = np.linspace(0, 1/f0, points_last_period)
        else:
            channel_data_last_period = txt_data
            time_modified = txt_data[:, 0]

        list_simulation_data = []
        list_return_dataset = []
        for count_var, variable in enumerate(variables):
            list_simulation_data.append(channel_data_last_period[:, count_var])

            if count_var != 0:
                list_return_dataset.append(Scope.generate_channel(
                    channel_time=time_modified, channel_data=list_simulation_data[count_var], channel_label=variable,
                    channel_source=channel_source, channel_unit=None, channel_linestyle=None, channel_color=None))

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Value of count_var {count_var}")

        return list_return_dataset

    @staticmethod
    def multiply(channel_voltage: 'Channel', channel_current: 'Channel', channel_label: Optional[str] = None) -> 'Channel':
        """
        calculate the power of two datasets.

        :param channel_voltage: dataset with voltage information
        :type channel_voltage: Channel
        :param channel_current: dataset with current information
        :type channel_current: Channel
        :param channel_label: label for new dataset_channel
        :type channel_label: str
        :return: power in a dataset
        :rtype: Channel
        """
        if not isinstance(channel_voltage, Channel):
            raise TypeError("channel_voltage must be type Scope.")
        if not isinstance(channel_current, Channel):
            raise TypeError("channel_current must be type Scope.")
        if not isinstance(channel_label, str) != channel_label is not None:
            raise TypeError("channel_label must be type str or None.")

        channel_data = channel_voltage.data * channel_current.data
        if channel_label is None and channel_voltage.label is not None \
                and channel_current.label is not None:
            channel_label = f"{channel_voltage.label} * {channel_current.label}"
        channel_power = Channel(channel_voltage.time, channel_data, label=channel_label,
                                unit='W', color=None, linestyle=None, source=None, modulename=class_modulename)

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of channel data elements={len(channel_data)}")

        return channel_power

    @staticmethod
    def integrate(channel_power: 'Channel', channel_label: Optional[str] = None):
        """
        Integrate a channels signal.

        The default use-case is calculating energy loss (variable naming is for the use case to calculate
        switch energy from power loss curve, e.g. from double-pulse measurement)

        :param channel_power: channel with power
        :type channel_power: Channel
        :param channel_label: channel label
        :type channel_label: Optional[str]
        :return: returns a scope-class, what integrates the input values
        :rtype: Channel
        """
        if not isinstance(channel_power, Channel):
            raise TypeError("channel_power must be type Scope.")
        if not isinstance(channel_label, str):
            raise TypeError("channel_label must be type str.")
        channel_energy = np.array([])
        timestep = channel_power.time[2] - channel_power.time[1]
        for count, _ in enumerate(channel_power.time):
            if count == 0:
                # set first energy value to zero
                channel_energy = np.append(channel_energy, 0)
            else:
                # using euler method
                energy = (np.nan_to_num(channel_power.data[count]) + np.nan_to_num(channel_power.data[count - 1])) / 2 * timestep
                channel_energy = np.append(channel_energy, channel_energy[-1] + energy)
        if channel_label is None:
            # Log missing user input
            logging.info(f"{class_modulename} :Label was not defined. So default value is used", class_modulename)
            channel_label = "Energy"

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of channel data elements={count}")

        return Channel(channel_power.time, channel_energy, label=channel_label, unit='J', color=None, source=None,
                       linestyle=None, modulename=class_modulename)

    @staticmethod
    def add(*channels: 'Channel') -> 'Channel':
        """
        Add channel_data of several Channels.

        :param channels: Input channels
        :type channels: Channel
        :return: Channel resulting from added input channels
        :rtype: Channel
        """
        if len(channels) < 2:
            raise ValueError("Minimum two channel inputs necessary!")

        # check input type and time data points
        for channel in channels:
            if not isinstance(channel, Channel):
                raise TypeError("channel must be type Scope.")
            if channel.time.all() != channels[0].time.all():
                raise ValueError("Can not add data. Different Channel.channel_time length!")
            if not (channel.time == channels[0].time).all():
                raise ValueError("Can not add data. Different Channel.channel_time values!")

        channel_data_result = np.zeros_like(channels[0].data)
        channel_label_result = ''
        for channel in channels:
            channel_data_result += channel.data
            channel_label_result += channel.label + ' + ' if channel.label is not None else ""
        channel_label_result = channel_label_result[:-3]

        # Log missing channel input, if amount of channels is one
        if len(channels) == 1:
            logging.info(f"{class_modulename} :Only on channel was provided. No channel was added.")
            # Log flow control

        logging.debug(f"{class_modulename} :FlCtl Amount of channels, which are added={len(channels)}")

        return Channel(time=channels[0].time, data=channel_data_result, unit=channels[0].unit,
                       label=channel_label_result, linestyle=None, color=None, source=None, modulename=class_modulename)

    @staticmethod
    def subtract(*channels: 'Channel') -> 'Channel':
        """
        Subtract channel_data of several Channels.

        :param channels: Input channels
        :type channels: Channel
        :return: Channel resulting from first input channel minus all following input channels
        :rtype: Channel
        """
        if len(channels) < 2:
            raise ValueError("Minimum two channel inputs necessary!")

        # check input type and time data points
        for channel in channels:
            if not isinstance(channel, Channel):
                raise TypeError("channel must be type Scope.")
            if channel.time.all() != channels[0].time.all():
                raise ValueError("Can not add data. Different Channel.channel_time length!")
            if not (channel.time == channels[0].time).all():
                raise ValueError("Can not add data. Different Channel.channel_time values!")

        channel_data_result = np.zeros_like(channels[0].data)
        channel_label_result = ''
        for channel_count, channel in enumerate(channels):
            if channel_count == 0:
                channel_data_result += channel.data
            else:
                channel_data_result -= channel.data
            channel_label_result += channel.label + ' - ' if channel.label is not None else ""
        channel_label_result = channel_label_result[:-3]

        # Log missing channel input, if amount of channels is one
        if len(channels) == 1:
            logging.info(f"{class_modulename} :Only on channel was provided. No channel was subtracted.")
            # Log flow control

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of channels, which are subtracted={len(channels)}")

        return Channel(channels[0].time, channel_data_result, unit=channels[0].unit,
                       label=channel_label_result, color=None, source=None, linestyle=None, modulename=class_modulename)

    @staticmethod
    def plot_channels(*channel: List['Channel'], timebase: str = 's', figure_size: Optional[Tuple] = None,
                      figure_directory: Optional[str] = None) -> plt.figure:
        """
        Plot channel datasets.

        :Examples:

        >>> import pysignalscope as pss
        >>> ch1, ch2, ch3, ch4 = pss.Scope.from_tektronix('tektronix_csv_file.csv')
        >>> pss.Scope.plot_channels([ch1, ch2, ch3],[ch4])

        Plots two subplots. First one has ch1, ch2, ch3, second one has ch4.

        Y-axis labels are set according to the channel_unit, presented in the last curve for the subplot.
        For own axis labeling, use as channel_unit for the last channel your label, e.g. r"$i_T$ in A".
        Note, that the r before the string gives the command to accept LaTeX formulas, like $$.

        :param channel: list of datasets
        :type channel: list[Channel]
        :param timebase: timebase, can be 's', 'ms', 'us', 'ns' or 'ps'
        :type timebase: str
        :param figure_size: None for auto-fit; fig_size for matplotlib (width, length in mm)
        :type figure_size: Tuple
        :param figure_directory: full path with file extension
        :type figure_directory: str

        :return: Plots
        :rtype: None
        """
        if timebase.lower() == 's':
            time_factor = 1.0
        elif timebase.lower() == 'ms':
            time_factor = 1e-3
        elif timebase.lower() == 'µs' or timebase.lower() == 'us':
            time_factor = 1e-6
            timebase = 'µs'
        elif timebase.lower() == 'ns':
            time_factor = 1e-9
        elif timebase.lower() == 'ps':
            time_factor = 1e-12
        else:
            time_factor = 1
            timebase = 's'

        if len(channel) == 1:  # This is for a single plot with multiple graphs
            fig = plt.figure(figsize=[x/25.4 for x in figure_size] if figure_size is not None else None, dpi=80)
            for plot_list in channel:
                count_legend_entries = 0
                for channel_dataset in plot_list:
                    plt.plot(channel_dataset.time / time_factor, channel_dataset.data,
                             label=channel_dataset.label, color=channel_dataset.color,
                             linestyle=channel_dataset.linestyle)
                    if channel_dataset.label is not None:
                        count_legend_entries += 1
                plt.grid()
                # plot legend in case of labels only. Otherwise, there would appear an empty box.
                if count_legend_entries != 0:
                    plt.legend()
                plt.xlabel(f'Time in {timebase}')
                if channel_dataset.unit is None:
                    pass
                elif channel_dataset.unit.lower() == 'v':
                    plt.ylabel(f"Voltage in {channel_dataset.unit}")
                elif channel_dataset.unit.lower() == 'a':
                    plt.ylabel(f"Current in {channel_dataset.unit}")
                elif channel_dataset.unit.lower() == 'w':
                    plt.ylabel(f"Power in {channel_dataset.unit}")
                else:
                    # in case of no matches, use a custom label. The channel_unit is used for this.
                    plt.ylabel(channel_dataset.unit)
            # Log flow control
            logging.debug(f"{class_modulename} :FlCtl Amount of plots within one channel={len(plot_list)}")

        else:  # This is for multiple plots with multiple graphs
            fig, axs = plt.subplots(nrows=len(channel), ncols=1, sharex=True, figsize=[x/25.4 for x in figure_size] if figure_size is not None else None)
            for plot_count, plot_list in enumerate(channel):
                count_legend_entries = 0
                for channel_dataset in plot_list:
                    axs[plot_count].plot(channel_dataset.time / time_factor, channel_dataset.data,
                                         label=channel_dataset.label, color=channel_dataset.color,
                                         linestyle=channel_dataset.linestyle)
                    if channel_dataset.label is not None:
                        count_legend_entries += 1
                axs[plot_count].grid()
                # plot legend in case of labels only. Otherwise, there would appear an empty box.
                if count_legend_entries != 0:
                    axs[plot_count].legend()
                axs[plot_count].set_xlabel(f'Time in {timebase}')
                if channel_dataset.unit is None:
                    pass
                elif channel_dataset.unit.lower() == 'v':
                    axs[plot_count].set_ylabel(f"Voltage in {channel_dataset.unit}")
                elif channel_dataset.unit.lower() == 'a':
                    axs[plot_count].set_ylabel(f"Current in {channel_dataset.unit}")
                elif channel_dataset.unit.lower() == 'w':
                    axs[plot_count].set_ylabel(f"Power in {channel_dataset.unit}")
                elif channel_dataset.unit.lower() == 'j':
                    axs[plot_count].set_ylabel(f"Energy in {channel_dataset.unit}")
                else:
                    # in case of no matches, use a custom label. The channel_unit is used for this.
                    axs[plot_count].set_ylabel(channel_dataset.unit)
            # Log flow control
            logging.debug(f"{class_modulename} :FlCtl Amount of plots within multiple channels={plot_count}")

        plt.tight_layout()
        if figure_directory is not None:
            plt.savefig(figure_directory, bbox_inches="tight")

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of channels, which are displayed={len(channel)}")

        plt.show()
        return fig

    @staticmethod
    def __check_limits(cur_value: float, min_value: float, max_value: float) -> bool:
        """
        Check if the  value is within the given range.

        :Example for a valid value:

        >>> bool valid
        >>> value = 10.2
        >>> valid = Scope.__check_limits(value, 3.2,11.3)
        >>> if valid:
        >>>     print(f"{value} is within the limit")
        >>> else:
        >>>     print(f"{value} is invalid")

        The  value will be check according the given limits.
        If the value is within the limit the method provide True as return value.

        :param cur_value: value to check
        :type cur_value: float
        :param min_value: lowest valid value
        :type min_value: float
        :param max_value: highest valid value
        :type max_value: float

        :return: condition result 'value is in range'
        :rtype: bool
        """
        # Init return variable
        ret_val = True

        # Check for maximum
        if cur_value > max_value:
            ret_val = False
        # Check for maximum
        if cur_value < min_value:
            ret_val = False

        return ret_val

    @staticmethod
    def __calculate_min_diff(cur_channel: np.array, ch_id: any) -> [bool, float]:
        """
        Check if the  value is within the given range.

        Calculate the minimal absolute differene of the values within the array (values will not be sorted).

        :Example for a valid value:

        >>> bool valid
        >>> channel5 = np.array([1, 2.4, 3.4, 4.4, 5])
        >>> valid, mindiff = Scope.__calculate_min_diff(channel5,5)
        >>> if valid:
        >>>     print(f"{mindiff} is the minimum difference")
        >>> else:
        >>>     print("Minimum difference could not be calculated")

        The minimum difference of a channel are calculated. A difference of 0 is ignored.
        The validity is set to false, if the array is not sorted in ascending order or the array contains only 1 value.

        :param cur_channel: value to check
        :type cur_channel: np.array
        :param ch_id: the lowest valid value
        :type ch_id: any

        :return: [Validity of the minimum value, minimum value]
        :rtype: bool,float
        """
        # Search minimal difference
        min_diff = 0
        validity = False

        # Check, if channel has got more, than one entry
        if cur_channel.size > 1:
            diff_channel = np.diff(cur_channel)
            if np.min(diff_channel) <= 0 or np.max(diff_channel) == 0:
                # Channel has got less equal on entry
                logging.warning(f"Channel {ch_id} is invalid (no ascending order or multiple values assigned to one coordinate")
            else:
                min_diff = np.min(diff_channel[diff_channel > 0])
                # Return value is valid
                validity = True
        else:
            # Channel has got less equal on entry
            logging.warning(f"Channel {ch_id} has got less equal on entry and will not be considered")

        # Return validity and minimal difference
        return [validity, min_diff]

    @staticmethod
    def plot_shiftchannels(channels: List['Channel'], shiftstep_x: Optional[float] = None, shiftstep_y: Optional[float] = None,
                           displayrange_x: Optional[Tuple[float, float]] = None, displayrange_y: Optional[Tuple[float, float]] = None) -> list[list[float]]:
        """
        Plot channel datasets.

        :Examples:

        >>> import pysignalscope as pss
        >>> ch1, ch2, ch3, ch4 = pss.Scope.from_tektronix('tektronix_csv_file.csv')
        >>> pss.Scope.plot_shiftchannels([ch1, ch2])

        Plots the channels ch1 and ch2. You can zoom into by selecting the zoom area with help of
        left mouse button. By moving the mouse while pressing the button  the area is marked by a red rectangle.
        If you release the left mouse button the area is marked. By moving the mouse within the area an perform
        a button press, you confirm and you zoom in the area. If you perform the left mouse button click outside
        of the marked area, you reject the selection. You reject the selection always by clicking the right mouse button independent you zoom out.
        button. If no area is selected or wait for confirmation, the click on the right mouse button leads to zooming out.
        There is a zoom limit in both directions. In this case, the rectangle shows the possible area (becomes larger),
        after you have release the left mouse button.

        Y-axis labels are set according to the channel_unit, presented in the last curve for the subplot.
        For own axis labeling, use as channel_unit for the last channel your label, e.g. r"$i_T$ in A".
        Note, that the r before the string gives the command to accept LaTeX formulas, like $$.
        The parameters has to fullfill conditions:
        Minimal shift step in x-direction is the minimal difference of 2 points of all provided channels

        :param channels: list of datasets
        :type channels: list[Channel]
        :param shiftstep_x: shift step in x-direction (optional parameter)
                            Has to be in range 'minimal difference of 2 points of the channels'
                            to ('displayed maximal x-value minus displayed minimal x-value')/10
        :type shiftstep_x: float
        :param shiftstep_y: shift step in y-direction (optional parameter)
                            Has to be in range ('displayed maximal y-value minus displayed minimal y-value')/200
                            to ('displayed maximal y-value minus displayed minimal y-value')/10
        :type shiftstep_y: float
        :param displayrange_x: Display range limits in x-direction (min_x, max_x)  (optional parameter)
                            Definition: delta_min_x = 100 * 'minimum distance between 2 samples', min_x = 'minimal x-value (of all channels)',
                            max_x = 'maximal x-value (of all channels)',  delta_x = max_x-min_x
                            The range for displayrange_x[0]: From min_x-delta_x to max_x-delta_min_x
                            The range for displayrange_x[1]: From min_x+delta_min_x to max_x+delta_x
                            and displayrange_x[1]-displayrange_x[0]>=delta_min_x
        :type displayrange_x: tuple of float
        :param displayrange_y: Display range limits in y-direction (min_y, max_y) (optional parameter)
                            Definition: delta_y = max_y-min_y, min_y = 'minimal y-value (of all channels)',
                            max_y = 'maximal y-value (of all channels)',  delta_min_y = delta_y/100
                            The range for displayrange_y[0]: From min_y-delta_y to max_y-delta_min_y*50
                            The range for displayrange_y[1]: From min_y+delta_min_y*50 to max_y-delta_y
                            and displayrange_y[1]-displayrange_y[0]>=delta_min_y*50
        :type displayrange_y: tuple of float

        :return: List of x and y-shifts per channel
        :rtype: list[list[float]]
        """
        # Init minimum and maximum values
        global_min_x = float(np.min(channels[0].time))
        global_max_x = np.max(channels[0].time)
        global_min_y = np.min(channels[0].data)
        global_max_y = np.max(channels[0].data)

        # For-loop over channels
        for channel in channels[1:]:
            global_min_x = np.min([global_min_x, np.min(channel.time)])
            global_max_x = np.max([global_max_x, np.max(channel.time)])
            global_min_y = np.min([global_min_y, np.min(channel.data)])
            global_max_y = np.max([global_max_y, np.max(channel.data)])

        # Search minimal difference
        min_diff_channel = 0
        # For-loop over channels to calculate the minimum distance between the values
        for channel_id, channel in enumerate(channels[1:], start=1):
            validity, min_diff = Scope.__calculate_min_diff(channel.time, channel_id)
            # Check, if the value is valid
            if validity:
                # Check, if a minimum is not set (min_diff_channel == 0
                if min_diff_channel == 0:
                    # First entry
                    min_diff_channel = min_diff
                else:
                    # Additional entries
                    min_diff_channel = np.min([min_diff_channel, min_diff])
            # Check if no channel provides a minimum
            if min_diff_channel == 0:
                # Invalid type of shift step x
                logging.error("Any channel has got invalid values (no ascending order or multiple values for x.")
                # Stop the program
                raise ValueError("Any channel has got invalid values (no ascending order or multiple values for x.")

        # Calculate max_shiftstepx as delta_max/10, min_shiftstepx as min_diff_channel  and
        # default value as delta_max/100
        delta_x = (global_max_x-global_min_x)
        # Check, if delta_x=0
        if delta_x == 0:
            delta_x = global_max_x
        # Set the shift steps in x-direction
        max_shiftstep_x = delta_x/10
        min_shiftstep_x = min_diff_channel
        def_shiftstep_x = max_shiftstep_x/50
        # Check, if default shift is less minimum shift
        if def_shiftstep_x < min_shiftstep_x:
            def_shiftstep_x = min_shiftstep_x

        # Calculate max_shiftstepy as delta_max/10, min_shiftstepx as delta_max/200  and
        # default value as delta_max/100
        delta_y = (global_max_y-global_min_y)
        # Check, if delta_y=0
        if delta_y == 0:
            delta_y = global_max_y
        # Set the shift steps in y-direction
        max_shiftstep_y = delta_y/10
        min_shiftstep_y = delta_y/200
        def_shiftstep_y = delta_y/100

        # Initialize values
        # Shift steps x
        if isinstance(shiftstep_x, float) or isinstance(shiftstep_x, int):
            if not Scope.__check_limits(shiftstep_x, min_shiftstep_x, max_shiftstep_x):
                shiftstep_x = def_shiftstep_x
                # Shift step in x-Direction is out of range
                logging.warning(f"{class_modulename} :Shift step in x-direction {shiftstep_x} is out of range. " 
                                f"The range isn from {min_shiftstep_x} to {max_shiftstep_x}")
        elif shiftstep_x is None:
            shiftstep_x = def_shiftstep_x
        else:
            # Invalid type of shift step x
            logging.error("Type of optional parameter 'shiftstep_x' has to be 'float'.")
            # Stop the program
            raise TypeError("Type of optional parameter 'shiftstep_x' has to be 'float'.")

        # Shift steps y
        if isinstance(shiftstep_y, float) or isinstance(shiftstep_y, int):
            if not Scope.__check_limits(shiftstep_y, min_shiftstep_y, max_shiftstep_y):
                shiftstep_y = def_shiftstep_y
                # Shift step in y-Direction is out of range
                logging.warning(f"{class_modulename} :Shift step in x-direction {shiftstep_x} is out of range. " 
                                f"The range isn from {min_shiftstep_x} to {max_shiftstep_x}")
                # logging
        elif shiftstep_y is None:
            shiftstep_y = def_shiftstep_y
        else:
            # Invalid type of shift step y
            logging.error("Type of optional parameter 'shiftstep_y' has to be 'float'.")
            # Stop the program
            raise TypeError("Type of optional parameter 'shiftstep_y' has to be 'float'.")

        # Initialize the actual display range y with an invalid value
        act_displayrange_x = [0.0, 0.0]
        # Evaluate display range x
        if isinstance(displayrange_x, tuple) and len(displayrange_x) == 2 \
           and isinstance(displayrange_x[0], (float, int)) and isinstance(displayrange_x[1], (float, int)):
            # Allow +-100 Percent: Calculate the delta
            global_delta = global_max_x - global_min_x
            display_delta = displayrange_x[1] - displayrange_x[0]
            if (displayrange_x[0] < global_min_x - global_delta) \
               or (displayrange_x[1] > global_max_x + global_delta) \
               or global_delta < (min_shiftstep_x * 5):
                # Display range in x-direction exceeds the limit
                logging.warning(
                    f"Display range in x-direction of min,max: {act_displayrange_x[0]},{act_displayrange_x[1]}  exceeds the limit "
                    f"min,max: {global_min_x - global_delta},{global_max_x + global_delta}.")
            elif display_delta < 100 * min_shiftstep_x:
                # Display range in x-direction exceeds the limit
                logging.warning(
                    f"Display range in x-direction of max-min: {display_delta} is to small (should be {100 * min_shiftstep_x})"
                    f"min,max: {global_min_x - global_delta},{global_max_x + global_delta}.")
            else:
                # Overtake display range in x-direction
                act_displayrange_x[0] = displayrange_x[0]
                act_displayrange_x[1] = displayrange_x[1]
        elif displayrange_x is not None:
            # Invalid type of Display range x
            logging.error("Type of optional parameter 'displayrange_x' has to be 'tuple[float][float]'.")
            # Stop the program
            raise TypeError("Type of optional parameter 'displayrange_x' has to be 'tuple[float][float]'.")

        # Initialize the actual display range y with an invalid value
        act_displayrange_y = [0.0, 0.0]
        # Evaluate display range y
        if isinstance(displayrange_y, tuple) and len(displayrange_y) == 2 \
           and isinstance(displayrange_y[0], (float, int)) and isinstance(displayrange_y[1], (float, int)):
            # Allow +-100 Percent: Calculate the delta
            global_delta = global_max_y - global_min_y
            display_delta = displayrange_y[1] - displayrange_y[0]
            if (displayrange_y[0] < (global_min_y - global_delta)) \
               or (displayrange_y[1] > (global_max_y + global_delta)) \
               or (global_delta < (min_shiftstep_y * 5)):
                # Display range in y-direction exceeds the limit
                logging.warning(
                    f"Display range in y-direction of min,max: {displayrange_y[0]},{displayrange_y[1]}  exceeds the limit "
                    f"min,max: {global_min_y - global_delta},{global_max_y + global_delta}.")
            elif display_delta < global_delta / 100:
                # Display range in y-direction exceeds the limit
                logging.warning(
                    f"Display range in y-direction of max-min: {display_delta} is to small (should be {global_delta / 100})"
                    f"min,max: {global_min_x - global_delta},{global_max_x + global_delta}.")
                # Set Display range in y-direction to max y-value
            else:
                # Overtake display range in y-direction
                act_displayrange_y[0] = displayrange_y[0]
                act_displayrange_y[1] = displayrange_y[1]
        elif displayrange_y is not None:
            # Invalid type of Display range y
            logging.error("Type of optional parameter 'displayrange_y' has to be 'tuple[float][float]'.")
            # Stop the program
            raise TypeError("Type of optional parameter 'displayrange_y' has to be 'tuple[float][float]'.")

        # Create instance variable
        ch_shift = scope_ch_shift()

        # Set the limits of shiftstep and display-range
        ch_shift.init_shiftstep_limits((min_shiftstep_x, max_shiftstep_x), (min_shiftstep_y, max_shiftstep_y))
        # Return the list of channel shifts
        return ch_shift.channel_shift(channels, shiftstep_x, shiftstep_y, act_displayrange_x, act_displayrange_y)

    @staticmethod
    def compare_channels(*channel_datasets: 'Channel', shift: Optional[List[Union[None, float]]] = None,
                         scale: Optional[List[Union[None, float]]] = None, offset: Optional[List[Union[None, float]]] = None,
                         timebase: str = 's'):
        """
        Graphical comparison for datasets. Note: Datasets need to be type Channel.

        :param channel_datasets: dataset according to Channel
        :type channel_datasets: Channel
        :param shift: phase shift in a list for every input dataset
        :type shift: list[Union[None, float]]
        :param scale: channel scale factor in a list for every input dataset
        :type scale: list[Union[None, float]]
        :param offset: channel offset in a list for every input dataset
        :type offset: list[Union[None, float]]
        :param timebase: timebase, can be 's', 'ms', 'us', 'ns' or 'ps'
        :type timebase: str
        """
        if timebase.lower() == 's':
            time_factor = 1.0
        elif timebase.lower() == 'ms':
            time_factor = 1e-3
        elif timebase.lower() == 'µs' or timebase.lower() == 'us':
            time_factor = 1e-6
            timebase = 'µs'
        elif timebase.lower() == 'ns':
            time_factor = 1e-9
        elif timebase.lower() == 'ps':
            time_factor = 1e-12
        else:
            # Log user info
            logging.info(f"{class_modulename} :time base was not defined or unknown (Set to {timebase}).\n timebase are set to second")

            time_factor = 1
            timebase = 's'

        for count, channel_dataset in enumerate(channel_datasets):
            if not isinstance(channel_dataset, Channel):
                raise TypeError("channel_dataset must be type Scope.")
            modified_time = channel_dataset.time
            modified_data = channel_dataset.data

            if shift is not None:
                modified_time = channel_dataset.time + shift[count]
            if scale is not None:
                modified_data = modified_data * scale[count]
            if offset is not None:
                modified_data = modified_data + offset[count]

            plt.plot(modified_time / time_factor, modified_data, label=channel_dataset.label,
                     color=channel_dataset.color, linestyle=channel_dataset.linestyle)
        plt.xlabel(f"time in {timebase}")
        if channel_datasets[0].unit is not None:
            if channel_datasets[0].unit.lower() == 'a':
                plt.ylabel('Current in A')
            elif channel_datasets[0].unit.lower() == 'u':
                plt.ylabel('Voltage in V')
            elif channel_datasets[0].unit.lower() == 'w':
                plt.ylabel('Power in W')
        plt.legend()
        plt.grid()
        plt.show()

        # Log flow control
        logging.debug(f"{class_modulename} :Amount of displayed datasets={len(channel_datasets)}")

    @staticmethod
    def fft(channel: Channel, plot: bool = True):
        """
        Perform fft to the signal.

        :param channel: Scope channel object
        :type channel: Channel
        :param plot: True to show a figure
        :type plot: bool
        :return: numpy-array [[frequency-vector],[amplitude-vector],[phase-vector]]
        :rtype: npt.NDArray[list]

        :Example:

        >>> import pysignalscope as pss
        >>> import numpy as np
        >>> channel_example = pss.Scope.from_numpy(np.array([[0, 5e-3, 10e-3, 15e-3, 20e-3], [1, -1, 1, -1, 1]]), f0=100000, mode='time')
        >>> pss.Scope.fft(channel_example)
        """
        if not isinstance(plot, bool):
            raise TypeError("plot must be type bool.")
        period_vector = np.array([channel.time, channel.data])

        # Log flow control
        logging.debug(f"{channel.modulename} :Amount of channel data={len(channel.data)}")

        return functions.fft(period_vector, mode='time', plot=plot)

    @staticmethod
    def short_to_period(channel: Channel, f0: Union[float, int, None] = None, time_period: Union[float, int, None] = None,
                        start_time: Union[float, int, None] = None):
        """Short a given Scope object to a period.

        :param channel: Scope channel object
        :type channel: Channel
        :param f0: frequency in Hz
        :type f0: float
        :param time_period: time period in seconds
        :type time_period: float
        :param start_time: start time in seconds
        :type start_time: float
        """
        if not isinstance(f0, (float, int)) != f0 is not None:
            raise TypeError("f0 must be type float/int/None.")
        if not isinstance(time_period, (float, int)) != f0 is not None:
            raise TypeError("time_period must be type float/int/None.")
        if not isinstance(start_time, (float, int)) != f0 is not None:
            raise TypeError("start_time must be type float/int/None.")

        if start_time is None:
            start_time = channel.time[0]
        # check for correct input parameter
        if time_period is None and f0 is None:
            raise ValueError("give a time period or a fundamental frequency")

        if time_period is not None:
            end_time = start_time + time_period
        elif f0 is not None:
            end_time = start_time + 1/f0
        channel = Scope.modify(channel, channel_time_cut_min=start_time, channel_time_cut_max=end_time)
        # Log flow control
        logging.debug(f"{channel.modulename} :Time range: {start_time} to {end_time}")

        return channel

    @staticmethod
    def low_pass_filter(channel: Channel, order: int = 1, angular_frequency_rad: float = 0.05) -> Channel:
        """
        Implement a butterworth filter on the given signal.

        See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

        :param channel: Channel object
        :type channel: Channel
        :param order: filter order
        :type order: int
        :param angular_frequency_rad: angular frequency in rad. Valid for values 0...1. Smaller value means lower filter frequency.
        :type angular_frequency_rad: float
        :return: Scope object with filtered channel_data
        :rtype: Channel
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be of type Scope.")
        if not isinstance(order, int):
            raise TypeError("order must be of type int.")
        if order <= 0:
            raise ValueError("minimum order is 1.")
        if not isinstance(angular_frequency_rad, float):
            raise TypeError("angular_frequency_rad must be of type float.")
        if angular_frequency_rad >= 1 or angular_frequency_rad <= 0:
            raise ValueError("angular_frequency_rad must be in interval ]0...1[.")

        # introduce scope copy for further channel modifications
        scope_copy = Scope.copy(channel)

        # filter adapted according to scipy example, see also:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
        b, a = signal.butter(order, angular_frequency_rad, btype="lowpass")
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, channel.data, zi=zi * channel.data[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
        y = signal.filtfilt(b, a, channel.data)

        # overwrite scope data of the copy
        scope_copy.data = y
        return scope_copy

    @staticmethod
    def derivative(channel: Channel, order: int = 1) -> Channel:
        """
        Get the derivative of the channel_data.

        In case of measured input signal, it is useful to apply a low-pass filter first.

        :param channel: Scope object
        :type channel: Channel
        :param order: oder of derivative, e.g. 1st order, ...
        :type order: int
        :return: Scope object
        :rtype: Channel
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be type Scope.")
        if not isinstance(order, int):
            raise TypeError("order must be type integer.")
        if order <= 0:
            raise ValueError("order must be > 0.")
        # make a copy of the input channel object
        channel_copy = Scope.copy(channel)

        # calculate the derivative, using findiff-toolbox
        d_dx = FinDiff(0, channel.time, order)
        df_dx = d_dx(channel.data)

        # apply the derivative to the scope channel copy
        channel_copy.data = df_dx

        return channel_copy

    @staticmethod
    def calc_rms(channel: Channel) -> Any:
        """
        Calculate the RMS of a given channel. Make sure to provide a SINGLE PERIOD of the signal.

        :param channel: Scope channel object
        :type channel: Scope
        :return: rms(self.channel_data).
        :rtype: Any
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be type Scope.")

        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.data)}")

        return np.sqrt(np.mean(channel.data ** 2))

    @staticmethod
    def calc_mean(channel: Channel) -> Any:
        """
        Calculate the mean of the given channel. Make sure to provide a SINGLE PERIOD of the signal.

        :param channel: Scope channel object
        :type channel: Scope
        :return: mean(self.channel_data)
        :rtype: any
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be type Scope.")

        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.data)}")

        return np.mean(channel.data)

    @staticmethod
    def calc_absmean(channel: Channel) -> Any:
        """
        Calculate the absolute mean of the given channel. Make sure to provide a SINGLE PERIOD of the signal.

        :param channel: Scope channel object
        :type channel: Scope
        :return: abs(mean(self.channel_data))
        :rtype: Any
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be type Scope.")

        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.data)}")

        return np.mean(np.abs(channel.data))

    @staticmethod
    def calc_abs(channel: Channel) -> Channel:
        """
        Modify the existing scope channel so that the signal is rectified.

        :param channel: Scope channel object
        :type channel: Scope
        :return: abs(channel.channel_data).
        :rtype: Scope
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be type Scope.")

        channel_modified = copy.deepcopy(channel)

        # Log flow control
        logging.debug(f"{channel_modified.modulename} :Number of channel data={len(channel_modified.data)}")

        channel_modified.data = np.abs(channel_modified.data)
        if channel_modified.label is not None:
            channel_modified.label = '|' + channel_modified.label + '|'

        return channel_modified

    @staticmethod
    def square(channel: Channel) -> Channel:
        """
        Square the data channel.

        :param channel: Scope channel object
        :type channel: Scope
        :return: channel.channel_data ** 2 as scope object
        :rtype: Scope
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be type Scope.")

        channel_modified = copy.deepcopy(channel)

        channel_modified.data = channel_modified.data ** 2
        if channel_modified.label is not None:
            channel_modified.label = channel_modified.label + '²'

        # Log flow control
        logging.debug(f"{channel_modified.modulename} :Number of channel data={len(channel_modified.data)}")

        return channel_modified

    @staticmethod
    def save_figure(figure: plt.figure, fig_name: str):
        """
        Save the given figure object as pdf.

        :param figure: figure object
        :type figure: matplotlib.pyplot.figure
        :param fig_name: figure name for pdf file naming
        :type fig_name: str
        """
        if isinstance(fig_name, str):
            figure.savefig(f"{fig_name}.pdf")
        else:
            raise TypeError("figure name must be of type str.")

        # Log flow control
        logging.debug(f"{class_modulename} :Name of file to save={fig_name}.pdf")

    @staticmethod
    def save(scope_object: Channel, filepath: str) -> None:
        """
        Save a scope object to hard disk.

        :param scope_object: scope object
        :type scope_object: Channel
        :param filepath: filepath including file name
        :type filepath: str
        """
        if not isinstance(filepath, str):
            raise TypeError("filepath must be of type str.")
        if ".pkl" not in filepath:
            filepath = filepath + ".pkl"
        file_path, file_name = os.path.split(filepath)
        if file_path == "":
            file_path = os.path.curdir
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        if not isinstance(scope_object, Channel):
            raise TypeError("scope_object must be of type Scope.")

        with open(filepath, 'wb') as handle:
            pickle.dump(scope_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath: str) -> Channel:
        """
        Load a scope file from the hard disk.

        :param filepath: filepath
        :type filepath: str
        :return: loaded Scope object
        :rtype: Channel
        """
        if not isinstance(filepath, str):
            raise TypeError("filepath must be of type str.")
        if ".pkl" not in filepath:
            raise ValueError("filepath must end with .pkl")
        if not os.path.exists(filepath):
            raise ValueError(f"{filepath} does not exist.")
        with open(filepath, 'rb') as handle:
            loaded_scope_object: Channel = pickle.load(handle)
        if not isinstance(loaded_scope_object, Channel):
            raise TypeError(f"Loaded object is of type {type(loaded_scope_object)}, but should be type Scope.")

        return loaded_scope_object


if __name__ == '__main__':
    pass

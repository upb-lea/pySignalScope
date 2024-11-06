"""Classes and methods to process scope data (from real scopes or from simulation tools) like in a real scope."""
# python libraries
import copy
import os.path
import warnings
import logging
from typing import Union, List, Tuple, Optional, Any

# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt
from lecroyutils.control import LecroyScope

# own libraries
import pysignalscope.functions as functions
from pysignalscope.logconfig import setup_logging
from pysignalscope.scope_dataclass import Scope

# - Logging setup ---------------------------------------------------------------------------------
setup_logging()

# Modul name für static methods
class_modulename = "TransformerCalculation"

# - Class definition ------------------------------------------------------------------------------

class HandleScope:
    """Class to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    @staticmethod
    def generate_scope_object(channel_time: Union[List[float], np.ndarray], channel_data: Union[List[float], np.ndarray],
                              channel_label: Optional[str] = None, channel_unit: Optional[str] = None, channel_color: Optional[str] = None,
                              channel_source: Optional[str] = None, channel_linestyle: Optional[str] = None) -> Scope:
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
        # check channel_label for a valid type
        if isinstance(channel_label, str) or channel_label is None:
            channel_label = channel_label
        else:
            raise TypeError("channel_label must be type str or None.")
        # check channel unit for a valid type
        if isinstance(channel_unit, str) or channel_unit is None:
            channel_unit = channel_unit
        else:
            raise TypeError("channel_unit must be type str or None.")
        # check channel_color for a valid type
        if isinstance(channel_color, str) or channel_color is None:
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

        return Scope(channel_time=channel_time,
                     channel_data=channel_data,
                     channel_label=channel_label,
                     channel_unit=channel_unit,
                     channel_color=channel_color,
                     channel_source=channel_source,
                     channel_linestyle=channel_linestyle,
                     modulename=class_modulename)

    # - Function modify ------------------------------------------------------------------------------

    @staticmethod
    def modify(channel: Scope, channel_data_factor: Optional[float] = None, channel_data_offset: Optional[float] = None,
               channel_label: Optional[str] = None, channel_unit: Optional[str] = None, channel_color: Optional[str] = None,
               channel_source: Optional[str] = None, channel_time_shift: Optional[float] = None,
               channel_time_shift_rotate: Optional[float] = None,
               channel_time_cut_min: Optional[float] = None, channel_time_cut_max: Optional[float] = None,
               channel_linestyle: Optional[str] = None) -> Scope:
        """
        Modify channel data like metadata or add a factor or offset to channel data.

        Useful for classes with channel_time/data, but without labels or units.

        :param channel: Scope channel object
        :type channel: Scope
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
        :rtype: Scope
        """
        if isinstance(channel_label, str) or channel_label is None:
            channel.channel_label = channel_label
            modify_flag = True
        else:
            raise TypeError("channel_label must be type str or None")
        if isinstance(channel_unit, str) or channel_unit is None:
            channel.channel_unit = channel_unit
            modify_flag = True
        else:
            raise TypeError("channel_unit must be type str or None")
        if isinstance(channel_data_factor, (int, float)):
            channel.channel_data = channel.channel_data * channel_data_factor
            modify_flag = True
        elif channel_data_factor is None:
            pass
        else:
            raise TypeError("channel_data_factor must be type float or None")
        if isinstance(channel_data_offset, (int, float)):
            channel.channel_data = channel.channel_data + channel_data_offset
            modify_flag = True
        elif channel_data_offset is None:
            pass
        else:
            raise TypeError("channel_data_offset must be type float or None")
        if isinstance(channel_color, str) or channel_color is None:
            channel.channel_color = channel_color
            modify_flag = True
        else:
            raise TypeError("channel_color must be type str or None")
        if isinstance(channel_source, str) or channel_source is None:
            channel.channel_source = channel_source
            modify_flag = True
        else:
            raise TypeError("channel_source must be type str or None")
        if isinstance(channel_time_shift, (int, float)):
            channel.channel_time = channel.channel_time + channel_time_shift
            modify_flag = True
        elif channel_time_shift is None:
            pass
        else:
            raise TypeError("channel_time_shift must be type float or None")
        if isinstance(channel_time_shift_rotate, (int, float)):
            # figure out current max time
            current_max_time = channel.channel_time[-1]
            current_period = current_max_time - channel.channel_time[0]
            # shift all times
            channel.channel_time = channel.channel_time + channel_time_shift_rotate
            channel.channel_time[channel.channel_time > current_max_time] = channel.channel_time[channel.channel_time > current_max_time] - current_period
            # due to rolling time-shift, channel_time and channel_data needs to be re-sorted.
            new_index = np.argsort(channel.channel_time)
            channel.channel_time = np.array(channel.channel_time)[new_index]
            channel.channel_data = np.array(channel.channel_data)[new_index]
            modify_flag = True
        elif channel_time_shift_rotate is None:
            pass
        else:
            raise TypeError("channel_time_shift_rotate must be type str or None")

        if isinstance(channel_time_cut_min, (int, float)):
            index_list_to_remove = []
            if channel_time_cut_min < channel.channel_time[0]:
                raise ValueError(f"channel_cut_time_min ({channel_time_cut_min}) < start of channel_time ({channel.channel_time[0]}). This is not allowed!")
            for count, value in enumerate(channel.channel_time):
                if value < channel_time_cut_min:
                    index_list_to_remove.append(count)
            channel.channel_time = np.delete(channel.channel_time, index_list_to_remove)
            channel.channel_data = np.delete(channel.channel_data, index_list_to_remove)
            modify_flag = True
        elif channel_time_cut_min is None:
            pass
        else:
            raise TypeError("channel_time_cut_min must be type float or None")

        if isinstance(channel_time_cut_max, (int, float)):
            index_list_to_remove = []
            if channel_time_cut_max > channel.channel_time[-1]:
                raise ValueError(f"channel_cut_time_max ({channel_time_cut_max}) > end of channel_time ({channel.channel_time[-1]}). This is not allowed!")
            for count, value in enumerate(channel.channel_time):
                if value > channel_time_cut_max:
                    index_list_to_remove.append(count)
            channel.channel_time = np.delete(channel.channel_time, index_list_to_remove)
            channel.channel_data = np.delete(channel.channel_data, index_list_to_remove)
            modify_flag = True
        elif channel_time_cut_max is None:
            pass
        else:
            raise TypeError("channel_time_cut_max must be type float or None")
        if isinstance(channel_linestyle, str):
            channel.channel_linestyle = channel_linestyle
            modify_flag = True
        elif channel_linestyle is None:
            pass
        else:
            raise TypeError("channel_linestyle must be type str or None")

        # Log flow control
        logging.debug(f"{channel.modulename} :FlCtl")
        # Log, if no modification is requested
        if not modify_flag:
            logging.info(f"{channel.modulename} : No modification is requested", )

        return channel

    @staticmethod
    def copy(channel: Scope):
        """Create a deepcopy of Channel."""
        return copy.deepcopy(channel)

    @staticmethod
    def from_tektronix(csv_file: str) -> List['Scope']:
        """
        Translate tektronix csv-file to a tuple of Channel.

        Note: Returns a tuple with four Channels (Tektronix stores multiple channel data in single .csv-file,
        this results to return of a tuple containing Channel's)

        :param csv_file: csv-file from tektronix scope
        :type csv_file: str
        :return: tuple of Channel, depending on the channel count stored in the .csv-file
        :rtype: list[Scope, Scope, Scope, Scope]

        :Example:

        >>> import pysignalscope as pss
        >>> [voltage, current_prim, current_sec] = pss.HandleScope.from_tektronix('/path/to/tektronix/file/tek0000.csv')
        """
        channel_source = 'Tektronix scope'

        if not isinstance(csv_file, str):
            raise TypeError("csv_file must be type str to show the full filepath.")

        file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=15)
        channel_counts = file.shape[1] - 1
        time = file[:, 0]

        channel_list = []
        for channel_count in range(1, channel_counts + 1):
            channel_list.append(Scope(channel_time=time,
                                      channel_data=file[:, channel_count],
                                      channel_source=channel_source,
                                      channel_label=None,
                                      channel_unit=None,
                                      channel_color=None,
                                      channel_linestyle=None,
                                      modulename=class_modulename))

        # Log user error Empty csv-file
        if channel_count == 0:
            logging.info(f"{class_modulename} : Invalid file or file without content")

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Channel counts {channel_counts}, File {csv_file}")

        return channel_list

    @staticmethod
    def from_tektronix_mso58(*csv_files: str) -> List['Scope']:
        """
        Translate tektronix csv-files to a list of Channel class objects.

        Note: insert multiple .csv-files to get a list of all channels.
        :param csv_files: csv-file from tektronix scope
        :type csv_files: str
        :return: List of Scope objects
        :rtype: List['Scope']

        :Example single channel:

        >>> import pysignalscope as pss
        >>> [current_prim] = pss.HandleScope.from_tektronix_mso58('/path/to/lecroy/files/current_prim.csv')

        :Example multiple channels channel:

        >>> import pysignalscope as pss
        >>> [current_prim, current_sec] = pss.HandleScope.from_tektronix_mso58('/path/one/current_prim.csv', '/path/two/current_sec.csv')
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

            tektronix_channels.append(Scope(time, ch1_data, channel_source=channel_source,
                                            channel_label=os.path.basename(csv_file).replace('.csv', ''),
                                            channel_unit=None, channel_linestyle=None, channel_color=None, modulename=class_modulename))

        # Log user error Empty csv-file
        if not tektronix_channels:
            logging.info(f"{class_modulename} : file {csv_files}-> Invalid file list or files without content")

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl {csv_files}")

        return tektronix_channels

    @staticmethod
    def from_tektronix_mso58_multichannel(csv_file: str) -> List['Scope']:
        """
        Translate tektronix csv-files to a list of Channel class objects.

        Note: insert multiple .csv-files to get a list of all channels.
        :param csv_file: csv-file from tektronix scope
        :type csv_file: str
        :return: List of Scope objects
        :rtype: List['Scope']

        :Example multiple channel csv-file:

        >>> import pysignalscope as pss
        >>> [current_prim, current_sec] = pss.HandleScope.from_tektronix_mso58_multichannel('/path/to/lecroy/files/currents.csv')
        """
        channel_source = 'Tektronix scope MSO58'

        if not isinstance(csv_file, str):
            raise TypeError("csv_file must be type str to show the full filepath.")

        file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=24)
        channel_counts = file.shape[1] - 1
        time = file[:, 0]

        channel_list = []
        for channel_count in range(1, channel_counts + 1):
            channel_list.append(Scope(time, file[:, channel_count], channel_source=channel_source, channel_color=None, channel_linestyle=None,
                                      channel_unit=None, channel_label=None, modulename=class_modulename))

        # Log user error Empty csv-file
        if channel_count == 0:
            logging.info(f"{class_modulename} : Invalid file or file without content", class_modulename)

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Channel counts {channel_count},File {csv_file}")

        return channel_list

    @staticmethod
    def from_lecroy(*csv_files: str) -> List['Scope']:
        """
        Translate LeCroy csv-files to a list of Channel class objects.

        Note: insert multiple .csv-files to get a list of all channels

        :param csv_files: csv-file from tektronix scope
        :type csv_files: str
        :return: List of scope objects
        :rtype: List['Scope']

        :Example single channel:

        >>> import pysignalscope as pss
        >>> [current_prim] = pss.HandleScope.from_lecroy('/path/to/lecroy/files/current_prim.csv')

        :Example multiple channels channel:

        >>> import pysignalscope as pss
        >>> [current_prim, current_sec] = pss.HandleScope.from_lecroy('/path/one/current_prim.csv', '/path/two/current_sec.csv')
        """
        channel_source = 'LeCroy scope'

        lecroy_channel = []
        for csv_file in csv_files:
            if not isinstance(csv_file, str):
                raise TypeError("csv_file must be type str to show the full filepath.")
            file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=5)
            time = file[:, 0]
            ch1_data = file[:, 1]

            lecroy_channel.append(Scope(channel_time=time, channel_data=ch1_data, channel_source=channel_source,
                                        channel_label=os.path.basename(csv_file).replace('.csv', ''),
                                        channel_unit=None, channel_color=None, channel_linestyle=None,
                                        modulename=class_modulename))

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
            return Scope(channel_time=data.x, channel_data=data.y, channel_source=channel_source, channel_label=channel_label,
                         channel_color=None, channel_linestyle=None, channel_unit=None, modulename=class_modulename)

    @staticmethod
    def from_numpy(period_vector_t_i: np.ndarray, mode: str = 'rad', f0: Union[float, None] = None,
                   channel_label=None, channel_unit=None) -> 'Scope':
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
        >>> channel = pss.HandleScope.from_numpy(np.array([[0, 5e-3, 10e-3, 15e-3, 20e-3], [1, -1, 1, -1, 1]]), f0=100000, mode='time')
        """
        # changes period_vector_t_i to a float array. e.g. the user inserts a vector like this
        # [[0, 90, 180, 270, 360], [1, -1, 1, -1, 1]], with degree-mode, there are only integers inside.
        # when re-assigning the vector by switching from degree to time, the floats will be changed to an integers,
        # what is for most high frequency signals a time-vector consisting with only zeros.
        period_vector_t_i = period_vector_t_i.astype(float)

        # check for correct input parameter
        if (mode == 'rad' or mode == 'deg') and f0 is None:
            raise ValueError("if mode is 'rad' or 'deg', a fundamental frequency f0 must be set")
        # check for input is list. Convert to numpy-array
        if isinstance(period_vector_t_i, List):
            period_vector_t_i = np.array(period_vector_t_i)

        # mode pre-calculation
        if mode == 'rad' and f0 is not None:
            period_vector_t_i[0] = period_vector_t_i[0] / (2 * np.pi * f0)
        elif mode == 'deg' and f0 is not None:
            period_vector_t_i[0] = period_vector_t_i[0] / (360 * f0)
        elif mode != 'time':
            raise ValueError("Mode not available. Choose: 'rad', 'deg', 'time'")

        single_dataset_channel = Scope(period_vector_t_i[0], period_vector_t_i[1],
                                       channel_label=channel_label, channel_unit=channel_unit, channel_color=None,
                                       channel_linestyle=None, modulename=class_modulename, channel_source=None)

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of Data: {len(single_dataset_channel)}")

        return single_dataset_channel

    @staticmethod
    def from_geckocircuits(txt_datafile: str, f0: Optional[float] = None) -> List['Scope']:
        """
        Convert a gecko simulation file to Channel.

        :param txt_datafile: path to text file, generated by geckoCIRCUITS
        :type txt_datafile: str
        :param f0: fundamental frequency [optional]
        :type f0: float
        :return: List of Channels
        :rtype: list[Scope]
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
                list_return_dataset.append(Scope(channel_time=time_modified, channel_data=list_simulation_data[count_var],
                                                 channel_label=variable, channel_source=channel_source, channel_unit=None,
                                                 channel_linestyle=None, channel_color=None, modulename=class_modulename))

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Value of count_var {count_var}")

        return list_return_dataset

    @staticmethod
    def power(channel_voltage: 'Scope', channel_current: 'Scope', channel_label: Optional[str] = None) -> 'Scope':
        """
        calculate the power of two datasets.

        :param channel_voltage: dataset with voltage information
        :type channel_voltage: Scope
        :param channel_current: dataset with current information
        :type channel_current: Scope
        :param channel_label: label for new dataset_channel
        :type channel_label: str
        :return: power in a dataset
        :rtype: Scope
        """
        if not isinstance(channel_voltage, Scope):
            raise TypeError("channel_voltage must be type Scope.")
        if not isinstance(channel_current, Scope):
            raise TypeError("channel_current must be type Scope.")
        if not isinstance(channel_label, str) != channel_label is not None:
            raise TypeError("channel_label must be type str or None.")

        channel_data = channel_voltage.channel_data * channel_current.channel_data
        if channel_label is None and channel_voltage.channel_label is not None \
                and channel_current.channel_label is not None:
            channel_label = f"{channel_voltage.channel_label} * {channel_current.channel_label}"
        channel_power = Scope(channel_voltage.channel_time, channel_data, channel_label=channel_label,
                              channel_unit='W', channel_color=None, channel_linestyle=None, channel_source=None, modulename=class_modulename)

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of channel data elements={len(channel_data)}")

        return channel_power

    @staticmethod
    def integrate(channel_power: 'Scope', channel_label: Optional[str] = None):
        """
        Integrate a channels signal.

        The default use-case is calculating energy loss (variable naming is for the use case to calculate
        switch energy from power loss curve, e.g. from double-pulse measurement)
        :param channel_power: channel with power
        :type channel_power: Scope
        :param channel_label: channel label
        :type channel_label: Optional[str]
        :return: returns a scope-class, what integrates the input values
        :rtype: Scope
        """
        if not isinstance(channel_power, Scope):
            raise TypeError("channel_power must be type Scope.")
        if not isinstance(channel_label, str):
            raise TypeError("channel_label must be type str.")
        channel_energy = np.array([])
        timestep = channel_power.channel_time[2] - channel_power.channel_time[1]
        for count, _ in enumerate(channel_power.channel_time):
            if count == 0:
                # set first energy value to zero
                channel_energy = np.append(channel_energy, 0)
            else:
                # using euler method
                energy = (np.nan_to_num(channel_power.channel_data[count]) + np.nan_to_num(channel_power.channel_data[count-1])) / 2 * timestep
                channel_energy = np.append(channel_energy, channel_energy[-1] + energy)
        if channel_label is None:
            # Log missing user input
            logging.info(f"{class_modulename} :Label was not defined. So default value is used", class_modulename)
            channel_label = "Energy"

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of channel data elements={count}")

        return Scope(channel_power.channel_time, channel_energy, channel_label=channel_label, channel_unit='J', channel_color=None, channel_source=None,
                     channel_linestyle=None, modulename=class_modulename)

    @staticmethod
    def add(*channels: 'Scope') -> 'Scope':
        """
        Add channel_data of several Channels.

        :param channels: Input channels
        :type channels: Scope
        :return: Channel resulting from added input channels
        :rtype: Scope
        """
        if len(channels) < 2:
            raise ValueError("Minimum two channel inputs necessary!")

        channel_data_result = np.zeros_like(channels[0].channel_data)
        channel_label_result = ''
        for channel in channels:
            if not isinstance(channel, Scope):
                raise TypeError("channel must be type Scope.")
            if channel.channel_time.all() != channels[0].channel_time.all():
                raise ValueError("Can not add data. Different Channel.channel_time length!")
            channel_data_result += channel.channel_data
            channel_label_result += channel.channel_label + ' + ' if channel.channel_label is not None else ""
        channel_label_result = channel_label_result[:-3]

        # Log missing channel input, if amount of channels is one
        if len(channels) == 1:
            logging.info(f"{class_modulename} :Only on channel was provided. No channel was added.")
            # Log flow control

        logging.debug(f"{class_modulename} :FlCtl Amount of channels, which are added={len(channels)}")

        return Scope(channel_time=channels[0].channel_time, channel_data=channel_data_result, channel_unit=channels[0].channel_unit,
                     channel_label=channel_label_result, channel_linestyle=None, channel_color=None, channel_source=None, modulename=class_modulename)

    @staticmethod
    def subtract(*channels: 'Scope') -> 'Scope':
        """
        Subtract channel_data of several Channels.

        :param channels: Input channels
        :type channels: Scope
        :return: Channel resulting from first input channel minus all following input channels
        :rtype: Scope
        """
        if len(channels) < 2:
            raise ValueError("Minimum two channel inputs necessary!")

        channel_data_result = np.zeros_like(channels[0].channel_data)
        channel_label_result = ''
        for channel_count, channel in enumerate(channels):
            if not isinstance(channel, Scope):
                raise TypeError("channel must be type Scope.")
            if channel.channel_time.all() != channels[0].channel_time.all():
                raise ValueError("Can not add data. Different Channel.channel_time length!")
            if channel_count == 0:
                channel_data_result += channel.channel_data
            else:
                channel_data_result -= channel.channel_data
            channel_label_result += channel.channel_label + ' - ' if channel.channel_label is not None else ""
        channel_label_result = channel_label_result[:-3]

        # Log missing channel input, if amount of channels is one
        if len(channels) == 1:
            logging.info(f"{class_modulename} :Only on channel was provided. No channel was subtracted.")
            # Log flow control

        # Log flow control
        logging.debug(f"{class_modulename} :FlCtl Amount of channels, which are subtracted={len(channels)}")

        return Scope(channels[0].channel_time, channel_data_result, channel_unit=channels[0].channel_unit,
                     channel_label=channel_label_result, channel_color=None, channel_source=None, channel_linestyle=None, modulename=class_modulename)

    @staticmethod
    def plot_channels(*channel: List['Scope'], timebase: str = 's', figure_size: Optional[Tuple] = None,
                      figure_directory: Optional[str] = None) -> plt.figure:
        """
        Plot channel datasets.

        Examples:
        >>> import pysignalscope as pss
        >>> ch1, ch2, ch3, ch4 = pss.HandleScope.from_tektronix('tektronix_csv_file.csv')
        >>> pss.HandleScope.plot_channels([ch1, ch2, ch3],[ch4])
        Plots two subplots. First one has ch1, ch2, ch3, second one has ch4.

        Y-axis labels are set according to the channel_unit, presented in the last curve for the subplot.
        For own axis labeling, use as channel_unit for the last channel your label, e.g. r"$i_T$ in A".
        Note, that the r before the string gives the command to accept LaTeX formulas, like $$.

        :param channel: list of datasets
        :type channel: list[Scope]
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
                for channel_dataset in plot_list:
                    plt.plot(channel_dataset.channel_time / time_factor, channel_dataset.channel_data,
                             label=channel_dataset.channel_label, color=channel_dataset.channel_color,
                             linestyle=channel_dataset.channel_linestyle)
                plt.grid()
                plt.legend()
                plt.xlabel(f'Time in {timebase}')
                if channel_dataset.channel_unit is None:
                    pass
                elif channel_dataset.channel_unit.lower() == 'v':
                    plt.ylabel(f"Voltage in {channel_dataset.channel_unit}")
                elif channel_dataset.channel_unit.lower() == 'a':
                    plt.ylabel(f"Current in {channel_dataset.channel_unit}")
                elif channel_dataset.channel_unit.lower() == 'w':
                    plt.ylabel(f"Power in {channel_dataset.channel_unit}")
                else:
                    # in case of no matches, use a custom label. The channel_unit is used for this.
                    plt.ylabel(channel_dataset.channel_unit)
            # Log flow control
            logging.debug(f"{class_modulename} :FlCtl Amount of plots within one channel={len(plot_list)}")

        else:  # This is for multiple plots with multiple graphs
            fig, axs = plt.subplots(nrows=len(channel), ncols=1, sharex=True, figsize=[x/25.4 for x in figure_size] if figure_size is not None else None)
            for plot_count, plot_list in enumerate(channel):
                for channel_dataset in plot_list:
                    axs[plot_count].plot(channel_dataset.channel_time / time_factor, channel_dataset.channel_data,
                                         label=channel_dataset.channel_label, color=channel_dataset.channel_color,
                                         linestyle=channel_dataset.channel_linestyle)
                axs[plot_count].grid()
                axs[plot_count].legend()
                axs[plot_count].set_xlabel(f'Time in {timebase}')
                if channel_dataset.channel_unit is None:
                    pass
                elif channel_dataset.channel_unit.lower() == 'v':
                    axs[plot_count].set_ylabel(f"Voltage in {channel_dataset.channel_unit}")
                elif channel_dataset.channel_unit.lower() == 'a':
                    axs[plot_count].set_ylabel(f"Current in {channel_dataset.channel_unit}")
                elif channel_dataset.channel_unit.lower() == 'w':
                    axs[plot_count].set_ylabel(f"Power in {channel_dataset.channel_unit}")
                elif channel_dataset.channel_unit.lower() == 'j':
                    axs[plot_count].set_ylabel(f"Energy in {channel_dataset.channel_unit}")
                else:
                    # in case of no matches, use a custom label. The channel_unit is used for this.
                    axs[plot_count].set_ylabel(channel_dataset.channel_unit)
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
    def scope2plot(csv_file, scope: str = 'tektronix', order: str = 'single', timebase: str = 's',
                   channel_units: Optional[List[str]] = None, channel_labels: Optional[List[str]] = None):
        """
        Plot the scope signal.

        :param csv_file: csv file-name
        :type csv_file: str
        :param scope: oscilloscope type
        :type scope: str
        :param order: 'single' [default] for all plots in single subplots, or 'multi' for subplots with one curve
        :type order: str
        :param timebase: timebase, can be 's', 'ms', 'us', 'ns' or 'ps'
        :type timebase: str
        :param channel_units: units in a list [unit_ch1, unit_ch2, unit_ch3, unit_ch4], e.g. ['A', 'A', 'V', 'Ohm']
        :type channel_units: list[str]
        :param channel_labels: channel labels in a list [label_ch1, label_ch2, label_ch3, label_ch4]
        :type channel_labels: list[str]
        """
        if scope.lower() == 'tektronix':
            channel_list = HandleScope.from_tektronix(csv_file)
        elif scope.lower() == 'lecroy':
            channel_list = HandleScope.from_lecroy(csv_file)
        else:
            # Log user warning
            logging.warning(f"{class_modulename} :Scope {scope} is unknown. Set to Tektronix scope", class_modulename, scope)
            # Display message
            warnings.warn('Can not detect scope type. Set to Tektronix scope', stacklevel=2)
            channel_list = HandleScope.from_tektronix(csv_file)

        if channel_units is not None:
            for channel_count, channel in enumerate(channel_list):
                channel = HandleScope.modify(channel, channel_unit=channel_units[channel_count])

        if channel_labels is not None:
            for channel_count, channel in enumerate(channel_list):
                channel = HandleScope.modify(channel, channel_label=channel_labels[channel_count])

        if order.lower().replace(" ", "") == 'single':
            HandleScope.plot_channels(channel_list, timebase=timebase)
        else:
            HandleScope.plot_channels([channel_list[0]], [channel_list[1]], [channel_list[2]], [channel_list[3]],
                                      timebase=timebase)

        # Log flow control
        logging.debug(f"{class_modulename} :Data of file {csv_file} are displayed (Type {scope})")

    @staticmethod
    def compare_channels(*channel_datasets: 'Scope', shift: Optional[List[Union[None, float]]] = None,
                         scale: Optional[List[Union[None, float]]] = None, offset: Optional[List[Union[None, float]]] = None,
                         timebase: str = 's'):
        """
        Graphical comparison for datasets. Note: Datasets need to be type Channel.

        :param channel_datasets: dataset according to Channel
        :type channel_datasets: Scope
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
            if not isinstance(channel_dataset, Scope):
                raise TypeError("channel_dataset must be type Scope.")
            modified_time = channel_dataset.channel_time
            modified_data = channel_dataset.channel_data

            if shift is not None:
                modified_time = channel_dataset.channel_time + shift[count]
            if scale is not None:
                modified_data = modified_data * scale[count]
            if offset is not None:
                modified_data = modified_data + offset[count]

            plt.plot(modified_time/time_factor, modified_data, label=channel_dataset.channel_label,
                     color=channel_dataset.channel_color, linestyle=channel_dataset.channel_linestyle)
        plt.xlabel(f"time in {timebase}")
        if channel_datasets[0].channel_unit is not None:
            if channel_datasets[0].channel_unit.lower() == 'a':
                plt.ylabel('Current in A')
            elif channel_datasets[0].channel_unit.lower() == 'u':
                plt.ylabel('Voltage in V')
            elif channel_datasets[0].channel_unit.lower() == 'w':
                plt.ylabel('Power in W')
        plt.legend()
        plt.grid()
        plt.show()

        # Log flow control
        logging.debug(f"{class_modulename} :Amount of displayed datasets={len(channel_datasets)}")

    @staticmethod
    def fft(channel: Scope, plot: bool = True):
        """
        Perform fft to the signal.

        :param channel: Scope channel object
        :type channel: Scope
        :param plot: True to show a figure
        :type plot: bool
        :return: numpy-array [[frequency-vector],[amplitude-vector],[phase-vector]]
        :rtype: npt.NDArray[list]

        :Example:

        >>> import pysignalscope as pss
        >>> import numpy as np
        >>> channel = pss.HandleScope.from_numpy(np.array([[0, 5e-3, 10e-3, 15e-3, 20e-3], [1, -1, 1, -1, 1]]), f0=100000, mode='time')
        >>> pss.HandleScope.fft(channel)
        """
        if not isinstance(plot, bool):
            raise TypeError("plot must be type bool.")
        period_vector = np.array([channel.channel_time, channel.channel_data])

        # Log flow control
        logging.debug(f"{channel.modulename} :Amount of channel data={len(channel.channel_data)}")

        return functions.fft(period_vector, mode='time', plot=plot)

    @staticmethod
    def short_to_period(channel: Scope, f0: Union[float, int, None] = None, time_period: Union[float, int, None] = None,
                        start_time: Union[float, int, None] = None):
        """Short a given Scope object to a period.

        :param channel: Scope channel object
        :type channel: Scope
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
            start_time = channel.channel_time[0]
        # check for correct input parameter
        if time_period is None and f0 is None:
            raise ValueError("give a time period or a fundamental frequency")

        if time_period is not None:
            end_time = start_time + time_period
        elif f0 is not None:
            end_time = start_time + 1/f0
        channel = HandleScope.modify(channel, channel_time_cut_min=start_time, channel_time_cut_max=end_time)
        # Log flow control
        logging.debug(f"{channel.modulename} :Time range: {start_time} to {end_time}")

        return channel

    @staticmethod
    def rms(channel: Scope) -> Any:
        """
        Calculate the RMS of a given channel. Make sure to provide a SINGLE PERIOD of the signal.

        Returns: rms(self.channel_data).
        """
        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.channel_data)}")

        return np.sqrt(np.mean(channel.channel_data ** 2))

    @staticmethod
    def mean(channel: Scope) -> Any:
        """
        Calculate the mean of the given channel. Make sure to provide a SINGLE PERIOD of the signal.

        Returns: mean(self.channel_data).
        """
        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.channel_data)}")

        return np.mean(channel.channel_data)

    @staticmethod
    def absmean(channel: Scope) -> Any:
        """
        Calculate the absolute mean of the given channel. Make sure to provide a SINGLE PERIOD of the signal.

        Returns: abs(mean(self.channel_data)).
        """
        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.channel_data)}")

        return np.mean(np.abs(channel.channel_data))

    @staticmethod
    def abs(channel: Scope) -> Scope:
        """
        Modify the existing scope channel so that the signal is rectified.

        Returns: abs(channel.channel_data).
        """
        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.channel_data)}")

        channel.channel_data = np.abs(channel.channel_data)
        if channel.channel_label is not None:
            channel.channel_label = '|' + channel.channel_label + '|'

        return channel

    @staticmethod
    def square(channel: Scope) -> Scope:
        """
        Square the data channel.

        Returns: channel.channel_data ** 2.
        """
        channel.channel_data = channel.channel_data ** 2
        if channel.channel_label is not None:
            channel.channel_label = channel.channel_label + '²'

        # Log flow control
        logging.debug(f"{channel.modulename} :Number of channel data={len(channel.channel_data)}")

        return channel

    @staticmethod
    def save(figure: plt.figure, fig_name: str):
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


if __name__ == '__main__':
    pass

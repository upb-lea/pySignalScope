"""Classes and methods to process scope data (from real scopes or from simulation tools) like in a real scope."""
import copy
import os.path

import numpy as np
import warnings
from matplotlib import pyplot as plt
from typing import Union, List, Tuple, Optional, Any
import pysignalscope.functions as functions


class Scope:
    """Class to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    def __init__(self, channel_time: Union[List[float], np.ndarray], channel_data: Union[List[float], np.ndarray],
                 channel_label: Optional[str] = None, channel_unit: Optional[str] = None, channel_color: Optional[str] = None,
                 channel_source: Optional[str] = None, channel_linestyle: Optional[str] = None) -> None:
        if isinstance(channel_time, List):
            self.channel_time = np.array(channel_time)
        elif isinstance(channel_time, np.ndarray):
            self.channel_time = channel_time
        else:
            raise Exception("channel_time must be type list or ArrayLike")
        if isinstance(channel_data, List):
            self.channel_data = np.array(channel_data)
        elif isinstance(channel_data, np.ndarray):
            self.channel_data = channel_data
        else:
            raise Exception("channel_data must be type list or ArrayLike")
        self.channel_label = channel_label
        self.channel_unit = channel_unit
        self.channel_color = channel_color
        self.channel_source = channel_source
        self.channel_linestyle = channel_linestyle

    def modify(self, channel_data_factor: Optional[float] = None, channel_data_offset: Optional[float] = None,
               channel_label: Optional[str] = None, channel_unit: Optional[str] = None, channel_color: Optional[str] = None,
               channel_source: Optional[str] = None, channel_time_shift: Optional[float] = None,
               channel_time_shift_rotate: Optional[float] = None,
               channel_time_cut_min: Optional[float] = None, channel_time_cut_max: Optional[float] = None,
               channel_linestyle: Optional[str] = None) -> None:
        """
        Modify channel data like metadata or add a factor or offset to channel data.

        Useful for classes with channel_time/data, but without labels or units.

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
        :return: None
        :rtype: None
        """
        if channel_label is not None:
            self.channel_label = channel_label
        if channel_unit is not None:
            self.channel_unit = channel_unit
        if channel_data_factor is not None:
            self.channel_data = self.channel_data * channel_data_factor
        if channel_data_offset is not None:
            self.channel_data = self.channel_data + channel_data_offset
        if channel_color is not None:
            self.channel_color = channel_color
        if channel_source is not None:
            self.channel_source = channel_source
        if channel_time_shift is not None:
            self.channel_time = self.channel_time + channel_time_shift
        if channel_time_shift_rotate is not None:
            # figure out current max time
            current_max_time = self.channel_time[-1]
            current_period = current_max_time - self.channel_time[0]
            # shift all times
            self.channel_time = self.channel_time + channel_time_shift_rotate
            self.channel_time[self.channel_time > current_max_time] = self.channel_time[self.channel_time > current_max_time] - current_period
            # due to rolling time-shift, channel_time and channel_data needs to be re-sorted.
            new_index = np.argsort(self.channel_time)
            self.channel_time = np.array(self.channel_time)[new_index]
            self.channel_data = np.array(self.channel_data)[new_index]

        if channel_time_cut_min is not None:
            index_list_to_remove = []
            if channel_time_cut_min < self.channel_time[0]:
                raise ValueError(f"channel_cut_time_min ({channel_time_cut_min}) < start of channel_time ({self.channel_time[0]}). This is not allowed!")
            for count, value in enumerate(self.channel_time):
                if value < channel_time_cut_min:
                    index_list_to_remove.append(count)
            self.channel_time = np.delete(self.channel_time, index_list_to_remove)
            self.channel_data = np.delete(self.channel_data, index_list_to_remove)

        if channel_time_cut_max is not None:
            index_list_to_remove = []
            if channel_time_cut_max > self.channel_time[-1]:
                raise ValueError(f"channel_cut_time_max ({channel_time_cut_max}) > end of channel_time ({self.channel_time[-1]}). This is not allowed!")
            for count, value in enumerate(self.channel_time):
                if value > channel_time_cut_max:
                    index_list_to_remove.append(count)
            self.channel_time = np.delete(self.channel_time, index_list_to_remove)
            self.channel_data = np.delete(self.channel_data, index_list_to_remove)
        if channel_linestyle is not None:
            self.channel_linestyle = channel_linestyle

    def copy(self):
        """Create a deepcopy of Channel."""
        return copy.deepcopy(self)

    @classmethod
    def from_tektronix(cls, csv_file: str) -> List['Scope']:
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
        >>> [voltage, current_prim, current_sec] = pss.Scope.from_tektronix('/path/to/tektronix/file/tek0000.csv')
        """
        channel_source = 'Tektronix scope'

        file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=15)
        channel_counts = file.shape[1] - 1
        time = file[:, 0]

        channel_list = []
        for channel_count in range(1, channel_counts + 1):
            channel_list.append(Scope(time, file[:, channel_count], channel_source=channel_source))

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
        >>> [current_prim] = pss.Scope.from_tektronix_mso58('/path/to/lecroy/files/current_prim.csv')

        :Example multiple channels channel:

        >>> import pysignalscope as pss
        >>> [current_prim, current_sec] = pss.Scope.from_tektronix_mso58('/path/one/current_prim.csv', '/path/two/current_sec.csv')
        """
        channel_source = 'Tektronix scope MSO58'

        tektronix_channels = []
        for csv_file in csv_files:
            file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=24)
            time = file[:, 0]
            ch1_data = file[:, 1]

            tektronix_channels.append(Scope(time, ch1_data, channel_source=channel_source,
                                            channel_label=os.path.basename(csv_file).replace('.csv', '')))

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
        >>> [current_prim, current_sec] = pss.Scope.from_tektronix_mso58_multichannel('/path/to/lecroy/files/currents.csv')
        """
        channel_source = 'Tektronix scope MSO58'

        file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=24)
        channel_counts = file.shape[1] - 1
        time = file[:, 0]

        channel_list = []
        for channel_count in range(1, channel_counts + 1):
            channel_list.append(Scope(time, file[:, channel_count], channel_source=channel_source))

        return channel_list

    @classmethod
    def from_lecroy(cls, *csv_files: str) -> List['Scope']:
        """
        Translate tektronix csv-files to a list of Channel class objects.

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
            file: np.ndarray = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=5)
            time = file[:, 0]
            ch1_data = file[:, 1]

            lecroy_channel.append(Scope(time, ch1_data, channel_source=channel_source,
                                        channel_label=os.path.basename(csv_file).replace('.csv', '')))

        return lecroy_channel

    @classmethod
    def from_numpy(cls, period_vector_t_i: np.ndarray, mode: str = 'rad', f0: Union[float, None] = None,
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
                                       channel_label=channel_label, channel_unit=channel_unit)

        return single_dataset_channel

    @classmethod
    def from_geckocircuits(cls, txt_datafile: str, f0: Optional[float] = None) -> List['Scope']:
        """
        Convert a gecko simulation file to Channel.

        :param txt_datafile: path to text file, generated by geckoCIRCUITS
        :type txt_datafile: str
        :param f0: fundamental frequency [optional]
        :type f0: float
        :return: List of Channels
        :rtype: list[Scope]
        """
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
                list_return_dataset.append(Scope(time_modified, list_simulation_data[count_var],
                                                 channel_label=variable, channel_source=channel_source))

        return list_return_dataset

    @classmethod
    def power(cls, channel_voltage: 'Scope', channel_current: 'Scope', channel_label: Optional[str] = None) -> 'Scope':
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
        channel_data = channel_voltage.channel_data * channel_current.channel_data
        if channel_label is None and channel_voltage.channel_label is not None \
                and channel_current.channel_label is not None:
            channel_label = f"{channel_voltage.channel_label} * {channel_current.channel_label}"
        channel_power = Scope(channel_voltage.channel_time, channel_data, channel_label=channel_label,
                              channel_unit='W')

        return channel_power

    @classmethod
    def integrate(cls, channel_power: 'Scope', channel_label: Optional[str] = None):
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
            channel_label = "Energy"
        return Scope(channel_power.channel_time, channel_energy, channel_label=channel_label, channel_unit='J')

    @classmethod
    def add(cls, *channels: 'Scope') -> 'Scope':
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
            if channel.channel_time.all() != channels[0].channel_time.all():
                raise ValueError("Can not add data. Different Channel.channel_time length!")
            channel_data_result += channel.channel_data
            channel_label_result += channel.channel_label + ' + ' if channel.channel_label is not None else ""
        channel_label_result = channel_label_result[:-3]

        return Scope(channels[0].channel_time, channel_data_result, channel_unit=channels[0].channel_unit,
                     channel_label=channel_label_result)

    @classmethod
    def subtract(cls, *channels: 'Scope') -> 'Scope':
        """
        Subtract channel_data of several Channels.

        :param channels: Input channels
        :type channels: Scope
        :return: Channel resulting from added input channels
        :rtype: Scope
        """
        if len(channels) < 2:
            raise ValueError("Minimum two channel inputs necessary!")

        channel_data_result = np.zeros_like(channels[0].channel_data)
        channel_label_result = ''
        for channel_count, channel in enumerate(channels):
            if channel.channel_time.all() != channels[0].channel_time.all():
                raise ValueError("Can not add data. Different Channel.channel_time length!")
            if channel_count == 0:
                channel_data_result += channel.channel_data
            else:
                channel_data_result -= channel.channel_data
            channel_label_result += channel.channel_label + ' - ' if channel.channel_label is not None else ""
        channel_label_result = channel_label_result[:-3]

        return Scope(channels[0].channel_time, channel_data_result, channel_unit=channels[0].channel_unit,
                     channel_label=channel_label_result)

    @classmethod
    def plot_channels(cls, *channel: List['Scope'], timebase: str = 's', figure_size: Optional[Tuple] = None,
                      figure_directory: Optional[str] = None) -> None:
        """
        Plot channel datasets.

        Examples:
        >>> import pysignalscope as pss
        >>> ch1, ch2, ch3, ch4 = pss.Scope.from_tektronix('tektronix_csv_file.csv')
        >>> pss.Scope.plot_channels([ch1, ch2, ch3],[ch4])
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
            plt.figure(figsize=[x/25.4 for x in figure_size] if figure_size is not None else None, dpi=80)
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
        plt.tight_layout()
        if figure_directory is not None:
            plt.savefig(figure_directory, bbox_inches="tight")
        plt.show()
        return fig

    @classmethod
    def scope2plot(cls, csv_file, scope: str = 'tektronix', order: str = 'single', timebase: str = 's',
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
            channel_list = Scope.from_tektronix(csv_file)
        elif scope.lower() == 'lecroy':
            channel_list = Scope.from_lecroy(csv_file)
        else:
            warnings.warn('Can not detect scope type. Set to Tektronix scope', stacklevel=2)
            channel_list = Scope.from_tektronix(csv_file)

        if channel_units is not None:
            for channel_count, channel in enumerate(channel_list):
                channel.modify(channel_unit=channel_units[channel_count])

        if channel_labels is not None:
            for channel_count, channel in enumerate(channel_list):
                channel.modify(channel_label=channel_labels[channel_count])

        if order.lower().replace(" ", "") == 'single':
            Scope.plot_channels(channel_list, timebase=timebase)
        else:
            Scope.plot_channels([channel_list[0]], [channel_list[1]], [channel_list[2]], [channel_list[3]],
                                timebase=timebase)

    @classmethod
    def compare_channels(cls, *channel_datasets: 'Scope', shift: Optional[List[Union[None, float]]] = None,
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
            time_factor = 1
            timebase = 's'

        for count, channel_dataset in enumerate(channel_datasets):
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

    def fft(self, plot: bool = True):
        """
        Perform fft to the signal.

        :param plot: True to show a figure
        :type plot: bool
        :return: numpy-array [[frequency-vector],[amplitude-vector],[phase-vector]]
        :rtype: npt.NDArray[list]

        :Example:

        >>> import pysignalscope as pss
        >>> import numpy as np
        >>> channel = pss.Scope.from_numpy(np.array([[0, 5e-3, 10e-3, 15e-3, 20e-3], [1, -1, 1, -1, 1]]), f0=100000, mode='time')
        >>> channel.fft()
        """
        period_vector = np.array([self.channel_time, self.channel_data])

        return functions.fft(period_vector, mode='time', plot=plot)

    def short_to_period(self, f0: Optional[float] = None, time_period: Optional[float] = None,
                        start_time: Optional[float] = None):
        """Short a given Scope object to a period.

        :param f0: frequency in Hz
        :type f0: float
        :param time_period: time period in seconds
        :type time_period: float
        :param start_time: start time in seconds
        :type start_time: float
        """
        if start_time is None:
            start_time = self.channel_time[0]
        # check for correct input parameter
        if time_period is None and f0 is None:
            raise ValueError("give a time period or a fundamental frequency")

        if time_period is not None:
            end_time = start_time + time_period
        elif f0 is not None:
            end_time = start_time + 1/f0
        self.modify(channel_time_cut_min=start_time, channel_time_cut_max=end_time)

    def plot(self, timebase: str = 's', figure_size: Optional[Tuple] = None, figure_directory: Optional[str] = None):
        """
        Plot a single scope channel.

        For multiple channel plots, use plot_channels.

        :param timebase: timebase, can be 's', 'ms', 'us', 'ns' or 'ps'
        :type timebase: str
        :param figure_size: None for auto-fit; fig_size for matplotlib (width, length)
        :type figure_size: Tuple
        :param figure_directory: full path with file extension
        :type figure_directory: str

        :return: Plots
        :rtype: None
        """
        self.plot_channels([self], timebase=timebase, figure_size=figure_size, figure_directory=figure_directory)

    def rms(self) -> Any:
        """
        Calculate the RMS of a given channel. Make sure to provide a SINGLE PERIOD of the signal.

        Returns: rms(self.channel_data).
        """
        return np.sqrt(np.mean(self.channel_data ** 2))

    def mean(self) -> Any:
        """
        Calculate the mean of the given channel. Make sure to provide a SINGLE PERIOD of the signal.

        Returns: mean(self.channel_data).
        """
        return np.mean(self.channel_data)

    def absmean(self) -> Any:
        """
        Calculate the absolute mean of the given channel. Make sure to provide a SINGLE PERIOD of the signal.

        Returns: abs(mean(self.channel_data)).
        """
        return np.mean(np.abs(self.channel_data))

    def abs(self) -> None:
        """
        Modify the existing scope channel so that the signal is rectified.

        Returns: abs(self.channel_data).
        """
        self.channel_data = np.abs(self.channel_data)
        if self.channel_label is not None:
            self.channel_label = '|' + self.channel_label + '|'

    def square(self) -> None:
        """
        Modify the existing scope channel so that the signal is squared.

        Returns: self.channel_data ** 2.
        """
        self.channel_data = self.channel_data ** 2
        if self.channel_label is not None:
            self.channel_label = self.channel_label + '²'

    @staticmethod
    def save(figure: plt.figure, fig_name: str):
        """
        Save the given figure object as pdf.

        :param figure: figure object
        :type figure: matplotlib.pyplot.figure
        :param fig_name: figure name for pdf file naming
        :type fig_name: str
        """
        figure.savefig(f"{fig_name}.pdf")


if __name__ == '__main__':
    pass

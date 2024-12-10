"""Generate, modify and plot Impedance objects."""
# python libraries
from typing import Union, List, Tuple, Optional
import copy
import cmath
import os
import pickle

# 3rd party libraries
import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt

# own libraries
from pysignalscope.impedance_dataclass import ImpedanceChannel

supported_measurement_devices = ['waynekerr', 'agilent']


class Impedance:
    """Class to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    @staticmethod
    def generate_impedance_object(frequency: Union[List, npt.ArrayLike], impedance: Union[List, npt.ArrayLike],
                                  phase: Union[List, npt.ArrayLike],
                                  label: Optional[str] = None, unit: Optional[str] = None, color: Union[str, tuple, None] = None,
                                  source: Optional[str] = None, linestyle: Optional[str] = None) -> ImpedanceChannel:
        """
        Generate the ImpedanceChannel object.

        :param frequency: channel frequency in Hz
        :type frequency: Union[List, npt.ArrayLike]
        :param impedance: channel impedance in Ohm
        :type impedance: Union[List, npt.ArrayLike]
        :param phase: channel phase in degree
        :type phase: Union[List, npt.ArrayLike]
        :param label: channel label to show in plots
        :type label: str
        :param unit: channel unit to show in plots
        :type unit: str
        :param color: channel color
        :type color: str
        :param source: Source, e.g. Measurement xy, Device yy
        :type source: str
        :param linestyle: line style for the plot e.g. '--'
        :type linestyle: str
        :return: ImpedanceChannel
        :rtype: ImpedanceChannel
        """
        if isinstance(frequency, List):
            frequency = np.array(frequency)
        elif isinstance(frequency, np.ndarray):
            frequency = frequency
        else:
            raise TypeError("frequency must be type list or ArrayLike")
        if isinstance(impedance, List):
            impedance = np.array(impedance)
        elif isinstance(impedance, np.ndarray):
            impedance = impedance
        else:
            raise TypeError("impedance must be type list or ArrayLike")
        if isinstance(phase, List):
            phase = np.array(phase)
        elif isinstance(phase, np.ndarray):
            phase = phase
        else:
            raise TypeError("phase must be type list or ArrayLike")

        # check for single non-allowed values in impedance
        if np.any(np.isnan(impedance)):
            raise ValueError("NaN is not allowed in impedance.")
        if np.any(np.isinf(impedance)):
            raise ValueError("inf is not allowed in impedance.")

        # check for single non-allowed values in phase
        if np.any(np.isnan(phase)):
            raise ValueError("NaN is not allowed in phase.")
        if np.any(np.isinf(phase)):
            raise ValueError("inf is not allowed in phase.")
        # check for empty data
        if frequency.size == 0:
            raise ValueError("Not allowed: frequency is empty")
        if impedance.size == 0:
            raise ValueError("Not allowed: impedance is empty")
        if phase.size == 0:
            raise ValueError("Not allowed: phase is empty")

        # check if frequency and impedance have the same length
        if len(frequency) != len(impedance):
            raise ValueError("frequency and impedance must be same length.")
        # check if frequency and phase have the same length
        if len(frequency) != len(phase):
            raise ValueError("frequency and phase must be same length.")

        # check if channel_time is strictly increasing
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("frequency not strictly increasing.")

        # check label for a valid type
        if isinstance(label, str) or label is None:
            label = label
        else:
            raise TypeError("label must be type str or None.")
        # check channel unit for a valid type
        if isinstance(unit, str) or unit is None:
            unit = unit
        else:
            raise TypeError("unit must be type str or None.")
        # check color for a valid type
        if isinstance(color, str) or color is None:
            color = color
        else:
            raise TypeError("color must be type str or None.")
        # check source for a valid type
        if isinstance(source, str) or source is None:
            source = source
        else:
            raise TypeError("source must be type str or None.")
        # check linestyle for a valid type
        if isinstance(linestyle, str) or linestyle is None:
            linestyle = linestyle
        else:
            raise TypeError("linestyle must be type str or None.")

        return ImpedanceChannel(
            frequency=frequency,
            impedance=impedance,
            phase_deg=phase,
            label=label,
            unit=unit,
            color=color,
            source=source,
            linestyle=linestyle)

    @staticmethod
    def modify(channel: ImpedanceChannel, impedance_factor: float = None, impedance_offset: float = None,
               label: str = None, unit: str = None, color: str = None,
               source: str = None, linestyle: str = None,
               frequency_cut_min: float = None, frequency_cut_max: float = None) -> ImpedanceChannel:
        """
        Modify channel data like metadata or add a factor or offset to channel data.

        Useful for ImpedanceChannel with frequency/data, but without labels or units.

        :param channel: Impedance object to modify
        :type channel: ImpedanceChannel
        :param impedance_factor: multiply channel.impedance by impedance_factor
        :type impedance_factor: float
        :param impedance_offset: add an offset to channel.impedance
        :type impedance_offset: float
        :param label: label to add to the Channel-class
        :type label: str
        :param unit: unit to add to the Channel-class
        :type unit: str
        :param color: Color of a channel
        :type color: str
        :param source: Source of a channel, e.g. 'GeckoCIRCUITS', 'Numpy', 'Tektronix-Scope', ...
        :type source: str
        :param frequency_cut_min: minimum frequency
        :type frequency_cut_min: float
        :param frequency_cut_max: maximum frequency
        :type frequency_cut_max: float
        :param linestyle: linestyle of channel, e.g. '--'
        :type linestyle: str
        :return: Modified ImpedanceChannel object
        :rtype: ImpedanceChannel
        """
        # deep copy to not modify the original input data
        modified_channel = copy.deepcopy(channel)

        if label is not None:
            modified_channel.label = label
        if unit is not None:
            modified_channel.unit = unit
        if impedance_factor is not None:
            modified_channel.impedance = modified_channel.impedance * impedance_factor
        if impedance_offset is not None:
            modified_channel.impedance = modified_channel.impedance + impedance_offset
        if color is not None:
            modified_channel.color = color
        if source is not None:
            modified_channel.source = source
        if linestyle is not None:
            modified_channel.linestyle = linestyle

        if frequency_cut_min is not None:
            index_list_to_remove = []
            for count, value in enumerate(modified_channel.frequency):
                if value < frequency_cut_min:
                    index_list_to_remove.append(count)
            modified_channel.frequency = np.delete(modified_channel.frequency, index_list_to_remove)
            modified_channel.impedance = np.delete(modified_channel.impedance, index_list_to_remove)
            modified_channel.phase_deg = np.delete(modified_channel.phase_deg, index_list_to_remove)

        if frequency_cut_max is not None:
            index_list_to_remove = []
            for count, value in enumerate(modified_channel.frequency):
                if value > frequency_cut_max:
                    index_list_to_remove.append(count)
            modified_channel.frequency = np.delete(modified_channel.frequency, index_list_to_remove)
            modified_channel.impedance = np.delete(modified_channel.impedance, index_list_to_remove)
            modified_channel.phase_deg = np.delete(modified_channel.phase_deg, index_list_to_remove)

        return modified_channel

    @staticmethod
    def copy(channel: ImpedanceChannel) -> ImpedanceChannel:
        """
        Create a deepcopy of ImpedanceChannel.

        :param channel: Impedance object
        :type channel: ImpedanceChannel
        :return: Deepcopy of the impedance object
        :rtype: ImpedanceChannel
        """
        if not isinstance(channel, ImpedanceChannel):
            raise TypeError("channel must be type ImpedanceChannel.")
        return copy.deepcopy(channel)

    @staticmethod
    def from_waynekerr(csv_filename: str, label: Optional[str] = None) -> 'ImpedanceChannel':
        """
        Bring csv-data from wayne kerr 6515b to Impedance.

        :param csv_filename: .csv filename from impedance analyzer
        :type csv_filename: str
        :param label: label to add to the Channel-class, optional.
        :type label: str
        :return: ImpedanceChannel object
        :rtype: ImpedanceChannel
        """
        impedance_measurement = np.genfromtxt(csv_filename, delimiter=',', dtype=float, skip_header=1,
                                              encoding='latin1')
        # move data to variables
        frequency = impedance_measurement[:, 0]
        impedance = impedance_measurement[:, 1]
        phase = impedance_measurement[:, 2]

        return Impedance.generate_impedance_object(frequency=frequency, impedance=impedance, phase=phase,
                                                   source='Impedance Analyzer Wayne Kerr 6515b', label=label)

    @staticmethod
    def from_kemet_ksim(csv_filename: str) -> 'ImpedanceChannel':
        """
        Import data from kemet "ksim" tool.

        :param csv_filename: path to csv-file
        :type csv_filename: str
        :return: ImpedanceChannel object
        :rtype: ImpedanceChannel
        """
        impedance_measurement = np.genfromtxt(csv_filename, delimiter=',', dtype=float, skip_header=1,
                                              encoding='latin1')
        # move data to variables
        frequency = impedance_measurement[:, 0]
        impedance = impedance_measurement[:, 3]
        phase = impedance_measurement[:, 4]

        return Impedance.generate_impedance_object(frequency=frequency, impedance=impedance, phase=phase,
                                                   source='https://ksim3.kemet.com/capacitor-simulation')

    @staticmethod
    def plot_impedance(channel_list: List, figure_size: Tuple = None) -> None:
        """
        Plot and compare impedance channels.

        :param channel_list: List with impedances
        :type channel_list: List
        :param figure_size: figure size as tuple in mm, e.g. (80, 80)
        :type figure_size: Tuple
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[x / 25.4 for x in figure_size] if figure_size is not None else None)
        for channel in channel_list:
            ax1.loglog(channel.frequency, channel.impedance, label=channel.label,
                       color=channel.color, linestyle=channel.linestyle)
            ax2.semilogx(channel.frequency, channel.phase_deg, label=channel.label,
                         color=channel.color, linestyle=channel.linestyle)

        ax1.grid()
        ax1.legend()
        ax1.set(xlabel='Frequency in Hz', ylabel=r'Impedance in $\Omega$')

        ax2.grid()
        ax2.set(xlabel='Frequency in Hz', ylabel='Phase in °')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_inductance_and_ac_resistance(channel_list: List) -> None:
        """
        Plot and compare inductance (in uH) and ac_resistance (Ohm) of impedance channels.

        :param channel_list: List with impedances
        :type channel_list: List
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        for channel in channel_list:
            ax1.semilogx(channel.frequency,
                         1e6 * channel.impedance * np.sin(np.deg2rad(channel.phase_deg)) / channel.frequency / 2 / np.pi,
                         label=channel.label, color=channel.color, linestyle=channel.linestyle)
            ax2.semilogx(channel.frequency,
                         channel.impedance * np.cos(np.deg2rad(channel.phase_deg)),
                         label=channel.label, color=channel.color, linestyle=channel.linestyle)

        ax1.grid()
        ax1.legend()
        ax1.set(xlabel='Frequency in Hz', ylabel=r'Inductance in $\rm{\mu}$H')

        ax2.grid()
        ax2.set(xlabel='Frequency in Hz', ylabel=r'AC Resistance in $\Omega$')
        ax2.legend()
        plt.show()

    @staticmethod
    def calc_re_im_parts(channel: ImpedanceChannel, show_figure: bool = True):
        """
        Calculate real and imaginary part of Impedance measurement.

        :param channel: Impedance object
        :type channel: ImpedanceChannel
        :param show_figure: Plot figure if true
        :type show_figure: bool
        :return: List with [(frequency, frequency_real_part), (frequency, frequency_imag_part)]
        :rtype: List
        """
        frequency_real_part = []
        frequency_imag_part = []
        complex_impedance = []

        for count_frequency, _ in enumerate(channel.frequency):
            impedance = channel.impedance[count_frequency] * np.exp(1j * channel.phase_deg[count_frequency] * 2 * np.pi / 360)
            complex_impedance.append(impedance)
            frequency_real_part.append(np.real(complex_impedance[count_frequency]))
            frequency_imag_part.append(np.imag(complex_impedance[count_frequency]))

        if show_figure:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax1.loglog(channel.frequency, frequency_real_part)
            ax1.grid()
            ax1.set_xlabel('frequency in Hz')
            ax1.set_ylabel('real (impedance)')

            ax2.loglog(channel.frequency, frequency_imag_part)
            ax2.set_xlabel('frequency in Hz')
            ax2.set_ylabel('imag (impedance)')
            ax2.grid()
            fig.show()

        frequency = copy.deepcopy(channel.frequency)

        return [(frequency, frequency_real_part), (frequency, frequency_imag_part)]

    @staticmethod
    def calc_rlc(channel: ImpedanceChannel, type_rlc: str, f_calc_c: float, f_calc_l: float, plot_figure: bool = False) -> tuple:
        """
        Calculate R, L, C values for given ImpedanceChannel.

        Calculated values will be drawn in a plot for comparison with the given data.

        :param channel: Impedance channel object
        :type channel: ImpedanceChannel
        :param type_rlc: Type 'R', 'L', 'C'
        :type type_rlc: str
        :param f_calc_c: Choose the frequency for calculation of C-value
        :type f_calc_c: float
        :param f_calc_l: Choose the frequency for calculation of L-value
        :type f_calc_l: float
        :param plot_figure: True/False [default] to plot the figure
        :type plot_figure: bool

        :return: Values for R, L, C
        :rtype: tuple

        :Example:

        >>> import pysignalscope as pss
        >>> example_data_rlc = pss.Impedance.from_rlc('l', 1000, 500e-6, 10e-12)
        >>> recalculated_r, recalculated_l, recalculated_c = pss.Impedance.calc_rlc(example_data_rlc, 'l', f_calc_l=10e3, f_calc_c=10e7, plot_figure=True)
        """
        # # Calculate R, L, C
        z_calc_c = np.interp(f_calc_c, channel.frequency, channel.impedance)
        z_calc_l = np.interp(f_calc_l, channel.frequency, channel.impedance)
        phase_calc_c = np.interp(f_calc_c, channel.frequency, channel.phase_deg)
        phase_calc_l = np.interp(f_calc_l, channel.frequency, channel.phase_deg)

        c_calc = 1 / (2 * np.pi * f_calc_c * z_calc_c)
        l_calc = z_calc_l / (2 * np.pi * f_calc_l)

        # # Calculate R at resonance frequency
        f_calc_r = 1 / (2 * np.pi * np.sqrt(l_calc * c_calc))
        z_calc_r = np.interp(f_calc_r, channel.frequency, channel.impedance)
        phase_calc_r = np.interp(f_calc_r, channel.frequency, channel.phase_deg)
        r_calc = z_calc_r

        if plot_figure:
            # Display calculated values
            print('C = {} F'.format(np.format_float_scientific(c_calc, precision=3)))
            print('R = {} Ohm'.format(np.format_float_scientific(r_calc, precision=3)))
            print('L = {} H'.format(np.format_float_scientific(l_calc, precision=3)))
            print('f_res = {} Hz'.format(np.format_float_scientific(f_calc_r, precision=1)))

            markerstyle = 'x'
            color_measurement = 'r'
            color_calculation = 'b'
            linestyle_measurement = '-'
            linestyle_calculation = ':'

            # plot output figure, compare measurement and interpolated data
            recalculated_curve = Impedance.from_rlc(type_rlc, r_calc, l_calc, c_calc)

            # generate plot
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

            # subplot 1 for impedance
            ax1.loglog(channel.frequency, channel.impedance, linestyle=linestyle_measurement, color=color_measurement, label='measurement')
            ax1.loglog(recalculated_curve.frequency, abs(recalculated_curve.impedance), linestyle=linestyle_calculation,
                       color=color_calculation, label='recalculated data')
            ax1.grid()
            ax1.legend()
            ax1.set(xlabel='Frequency in Hz', ylabel=r'Impedance in $\Omega$')
            ax1.plot(f_calc_c, z_calc_c, marker=markerstyle, color=color_measurement)
            ax1.plot(f_calc_r, z_calc_r, marker=markerstyle, color=color_measurement)
            ax1.plot(f_calc_l, z_calc_l, marker=markerstyle, color=color_measurement)

            # subplot 2 for phase
            ax2.semilogx(channel.frequency, channel.phase_deg, linestyle=linestyle_measurement, color=color_measurement, label='measurement')
            ax2.semilogx(recalculated_curve.frequency, recalculated_curve.phase_deg, linestyle=linestyle_calculation, color=color_calculation,
                         label='recalculated data')
            ax2.grid()
            ax2.set(xlabel='Frequency in Hz', ylabel='Phase in degree')
            ax2.legend()
            ax2.plot(f_calc_c, phase_calc_c, marker=markerstyle, color=color_measurement)
            ax2.plot(f_calc_r, phase_calc_r, marker=markerstyle, color=color_measurement)
            ax2.plot(f_calc_l, phase_calc_l, marker=markerstyle, color=color_measurement)
            plt.show()

        # return results
        return r_calc, l_calc, c_calc

    @staticmethod
    def from_rlc(type_rlc: str, resistance: Union[float, np.array], inductance: Union[float, np.array],
                 capacitance: Union[float, np.array]) -> 'ImpedanceChannel':
        """
        Calculate the impedance over frequency for R - L - C - combination.

        :param type_rlc: Type of network, can be 'R', 'L' or 'C'
        :type type_rlc: str
        :param resistance: resistance
        :type resistance: float
        :param inductance: inductance
        :type inductance: float
        :param capacitance: capacitance
        :type capacitance: bool
        :return: Impedance object
        :rtype: ImpedanceChannel

        :Example:

        >>> import pysignalscope as pss
        >>> impedance_channel_object = pss.Impedance.from_rlc('C', 10e-3, 100e-9, 36e-3)

         *  Type C and RLC

         .. code-block::

            ---R---L---C---

         *  Type R

         .. code-block::

            ---+---R---L---+---
               |           |
               +-----C-----+

         *  Type L

         .. code-block::

            ---+---L---+---
               |       |
               +---C---+
               |       |
               +---R---+
        """
        f = np.logspace(0, 8, 10000)

        if 'C' in type_rlc.upper() or 'RLC' in type_rlc.upper():
            z_total = resistance + 1j * 2 * np.pi * f * inductance - 1j / (2 * np.pi * f * capacitance)
        elif 'R' in type_rlc.upper():
            z_total = 1. / (1. / (resistance + 1j * 2 * np.pi * f * inductance) + 1j * 2 * np.pi * f * capacitance)
        elif 'L' in type_rlc.upper():
            z_total = 1 / (1. / resistance + 1. / (2 * np.pi * f * inductance * 1j) + 1j * 2 * np.pi * f * capacitance)
        else:
            raise ValueError("check input type_rlc!")

        phase_vector = []
        for i in z_total:
            phase_vector.append(cmath.phase(i) * 360 / (2 * np.pi))

        return Impedance.generate_impedance_object(frequency=f, impedance=np.abs(z_total), phase=phase_vector)

    @staticmethod
    def check_capacitor_from_waynekerr(csv_filename: str, label: str,
                                       target_capacitance: float, plot_figure: bool = True) -> 'ImpedanceChannel':
        """
        Check a capacitor impedance .csv-curve against a target capacitance.

        Reads the .csv-curve from wayne kerr impedance analyzer, calculates r, l, and c from the measurement
        and shows the derivation from the target capacitance. Visualizes the curves in case of plot_figure is set
        to true.

        :param csv_filename: filepath to Wayne Kerr impedance analyzer .csv-file
        :type csv_filename: str
        :param label: channel label for the plot
        :type label: str
        :param target_capacitance: target capacitance in F
        :type target_capacitance: float
        :param plot_figure: Set to True for plot
        :type plot_figure: bool

        :return: measured capacitor as impedance curve
        :rtype: ImpedanceChannel
        """
        f_calc_c = 100
        f_calc_l = 15e6

        capacitor_impedance = Impedance.from_waynekerr(csv_filename, label)

        r_measure, l_measure, c_measure = Impedance.calc_rlc(capacitor_impedance, "c", f_calc_c, f_calc_l, plot_figure=plot_figure)
        derivation_capacitance = c_measure / target_capacitance

        print(f"{derivation_capacitance=}")

        return capacitor_impedance

    @staticmethod
    def save(channel: ImpedanceChannel, filepath: str) -> None:
        """
        Save an ImpedanceChannel object to hard disk.

        :param channel: impedance channel
        :type channel: ImpedanceChannel
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
        if not isinstance(channel, ImpedanceChannel):
            raise TypeError("channel must be of type ImpedanceChannel.")

        with open(filepath, 'wb') as handle:
            pickle.dump(channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath: str) -> ImpedanceChannel:
        """
        Load an ImpedanceChannel file from the hard disk.

        :param filepath: filepath
        :type filepath: str
        :return: loaded ImpedanceChannel object
        :rtype: ImpedanceChannel
        """
        if not isinstance(filepath, str):
            raise TypeError("filepath must be of type str.")
        if ".pkl" not in filepath:
            raise ValueError("filepath must end with .pkl")
        if not os.path.exists(filepath):
            raise ValueError(f"{filepath} does not exist.")
        with open(filepath, 'rb') as handle:
            loaded_scope_object: ImpedanceChannel = pickle.load(handle)
        if not isinstance(loaded_scope_object, ImpedanceChannel):
            raise TypeError(f"Loaded object is of type {type(loaded_scope_object)}, but should be type Scope.")

        return loaded_scope_object

    @staticmethod
    def to_inductance(channel: ImpedanceChannel) -> ImpedanceChannel:
        """
        Convert an ImpedanceChannel to a pure inductance ImpedanceChannel.

        Ignore the resistive and capacitive part by trigonometric operation.

        :param channel: Impedance channel object
        :type channel: ImpedanceChannel
        :return: Impedance channel as inductance
        :rtype: ImpedanceChannel
        """
        if not isinstance(channel, ImpedanceChannel):
            raise TypeError("channel must be type ImpedanceChannel.")
        inductance_phase = 90 * np.ones_like(channel.phase_deg)
        inductance_impedance = channel.impedance * np.sin(np.deg2rad(channel.phase_deg))

        inductance_channel = Impedance.generate_impedance_object(channel.frequency, impedance=inductance_impedance, phase=inductance_phase)

        return inductance_channel

    @staticmethod
    def to_resistance(channel: ImpedanceChannel) -> ImpedanceChannel:
        """
        Convert an ImpedanceChannel to a pure resistive ImpedanceChannel.

        Ignore the inductive and capacitive part by trigonometric operation.

        :param channel: Impedance channel object
        :type channel: ImpedanceChannel
        :return: Impedance channel as inductance
        :rtype: ImpedanceChannel
        """
        if not isinstance(channel, ImpedanceChannel):
            raise TypeError("channel must be type ImpedanceChannel.")
        resistance_phase = np.zeros_like(channel.phase_deg)
        resistance_impedance = channel.impedance * np.cos(np.deg2rad(channel.phase_deg))

        resistance_channel = Impedance.generate_impedance_object(channel.frequency, impedance=resistance_impedance, phase=resistance_phase)

        return resistance_channel

    @staticmethod
    def to_capacitance(channel: ImpedanceChannel) -> ImpedanceChannel:
        """
        Convert an ImpedanceChannel to a pure capacitance ImpedanceChannel.

        Ignore the resistive and inductive part by trigonometric operation.

        :param channel: Impedance channel object
        :type channel: ImpedanceChannel
        :return: Impedance channel as inductance
        :rtype: ImpedanceChannel
        """
        if not isinstance(channel, ImpedanceChannel):
            raise TypeError("channel must be type ImpedanceChannel.")
        capacitance_phase = -90 * np.ones_like(channel.phase_deg)
        capacitance_impedance = -channel.impedance * np.sin(np.deg2rad(channel.phase_deg))

        capacitance_channel = Impedance.generate_impedance_object(channel.frequency, impedance=capacitance_impedance, phase=capacitance_phase)

        return capacitance_channel

    @staticmethod
    def plot_component(channel: ImpedanceChannel, figure_size: Optional[Tuple] = None) -> None:
        """
        Plot the component values.

        Phase must be -90°, 0°, +90° for all entries.

        :param channel: Impedance
        :type channel: List
        :param figure_size: figure size as tuple in mm, e.g. (80, 80)
        :type figure_size: Tuple
        """
        if not isinstance(channel, ImpedanceChannel):
            raise TypeError("channel must be type ImpedanceChannel.")

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[x / 25.4 for x in figure_size] if figure_size is not None else None)

        # type resistor
        if np.array_equal(channel.phase_deg, np.zeros_like(channel.phase_deg)):
            ax1.semilogx(channel.frequency, channel.impedance)
            ax1.set_ylabel(r"Resistance / $\Omega$")

        # type capacitance
        elif np.array_equal(channel.phase_deg, -90 * np.ones_like(channel.phase_deg)):
            capacitance = 1 / channel.impedance / 2 / np.pi / channel.frequency
            capacitance[capacitance <= 0] = np.nan
            ax1.semilogx(channel.frequency, capacitance)
            ax1.set_ylabel(r"Capacitance / F")

        # type inductance
        elif np.array_equal(channel.phase_deg, 90 * np.ones_like(channel.phase_deg)):
            inductance = channel.impedance / 2 / np.pi / channel.frequency
            inductance[inductance <= 0] = np.nan
            ax1.semilogx(channel.frequency, inductance)
            ax1.set_ylabel(r"Inductance / H")
        else:
            raise ValueError("Can not detect type of component. Phase must be -90°, 0°, +90° for all entries.")

        ax1.set_xlabel("Frequency / Hz")
        ax1.grid()
        ax2.semilogx(channel.frequency, channel.phase_deg)
        ax2.grid()
        ax2.set_xlabel("Frequency / Hz")
        ax2.set_ylabel("Phase / °")

        plt.show()

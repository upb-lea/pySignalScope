"""Generate, modify and plot Impedance objects."""
# python libraries
from typing import Union, List, Tuple, Optional
import copy
import cmath

# 3rd party libraries
import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt

# own libraries
from pysignalscope.impedance_dataclass import Impedance

supported_measurement_devices = ['waynekerr', 'agilent']


class HandleImpedance:
    """Class to share scope figures in a special format, to keep labels, units and voltages belonging to a certain curve."""

    @staticmethod
    def generate_impedance_object(channel_frequency: Union[List, npt.ArrayLike], channel_impedance: Union[List, npt.ArrayLike],
                                  channel_phase: Union[List, npt.ArrayLike],
                                  channel_label: str = None, channel_unit: str = None, channel_color: str = None,
                                  channel_source: str = None, channel_linestyle: str = None) -> Impedance:
        """
        Generate the impedance object.

        :param channel_frequency: channel frequency in Hz
        :type channel_frequency: Union[List, npt.ArrayLike]
        :param channel_impedance: channel impedance in Ohm
        :type channel_impedance: Union[List, npt.ArrayLike]
        :param channel_phase: channel phase in degree
        :type channel_phase: Union[List, npt.ArrayLike]
        :param channel_label: channel label to show in plots
        :type channel_label: str
        :param channel_unit: channel unit to show in plots
        :type channel_unit: str
        :param channel_color: channel color
        :type channel_color: str
        :param channel_source: Source, e.g. Measurement xy, Device yy
        :type channel_source: str
        :param channel_linestyle: line style for the plot e.g. '--'
        :type channel_linestyle: str
        """
        if isinstance(channel_frequency, List):
            channel_frequency = np.array(channel_frequency)
        elif isinstance(channel_frequency, np.ndarray):
            channel_frequency = channel_frequency
        else:
            raise TypeError("channel_frequency must be type list or ArrayLike")
        if isinstance(channel_impedance, List):
            channel_impedance = np.array(channel_impedance)
        elif isinstance(channel_impedance, np.ndarray):
            channel_impedance = channel_impedance
        else:
            raise TypeError("channel_impedance must be type list or ArrayLike")
        if isinstance(channel_phase, List):
            channel_phase = np.array(channel_phase)
        elif isinstance(channel_phase, np.ndarray):
            channel_phase = channel_phase
        else:
            raise TypeError("channel_phase must be type list or ArrayLike")

        return Impedance(
            channel_frequency=channel_frequency,
            channel_impedance=channel_impedance,
            channel_phase=channel_phase,
            channel_label=channel_label,
            channel_unit=channel_unit,
            channel_color=channel_color,
            channel_source=channel_source,
            channel_linestyle=channel_linestyle)

    @staticmethod
    def modify(channel: Impedance, channel_impedance_factor: float = None, channel_impedance_offset: float = None,
               channel_label: str = None, channel_unit: str = None, channel_color: str = None,
               channel_source: str = None, channel_linestyle: str = None,
               channel_frequency_cut_min: float = None, channel_frequency_cut_max: float = None) -> Impedance:
        """
        Modify channel data like metadata or add a factor or offset to channel data.

        Useful for classes with channel_frequency/data, but without labels or units.

        :param channel: Impedance object to modify
        :type channel: Impedance
        :param channel_impedance_factor: multiply channel.channel_impedance by channel_impedance_factor
        :type channel_impedance_factor: float
        :param channel_impedance_offset: add an offset to channel.channel_impedance
        :type channel_impedance_offset: float
        :param channel_label: label to add to the Channel-class
        :type channel_label: str
        :param channel_unit: unit to add to the Channel-class
        :type channel_unit: str
        :param channel_color: Color of a channel
        :type channel_color: str
        :param channel_source: Source of a channel, e.g. 'GeckoCIRCUITS', 'Numpy', 'Tektronix-Scope', ...
        :type channel_source: str
        :param channel_frequency_cut_min: minimum frequency
        :type channel_frequency_cut_min: float
        :param channel_frequency_cut_max: maximum frequency
        :type channel_frequency_cut_max: float
        :param channel_linestyle: linestyle of channel, e.g. '--'
        :type channel_linestyle: str
        :return: Modified impedance object
        :rtype: Impedance
        """
        # deep copy to not modify the original input data
        modified_channel = copy.deepcopy(channel)

        if channel_label is not None:
            modified_channel.channel_label = channel_label
        if channel_unit is not None:
            modified_channel.channel_unit = channel_unit
        if channel_impedance_factor is not None:
            modified_channel.channel_impedance = modified_channel.channel_impedance * channel_impedance_factor
        if channel_impedance_offset is not None:
            modified_channel.channel_impedance = modified_channel.channel_impedance + channel_impedance_offset
        if channel_color is not None:
            modified_channel.channel_color = channel_color
        if channel_source is not None:
            modified_channel.channel_source = channel_source
        if channel_linestyle is not None:
            modified_channel.channel_linestyle = channel_linestyle

        if channel_frequency_cut_min is not None:
            index_list_to_remove = []
            for count, value in enumerate(modified_channel.channel_frequency):
                if value < channel_frequency_cut_min:
                    index_list_to_remove.append(count)
            modified_channel.channel_frequency = np.delete(modified_channel.channel_frequency, index_list_to_remove)
            modified_channel.channel_impedance = np.delete(modified_channel.channel_impedance, index_list_to_remove)
            modified_channel.channel_phase = np.delete(modified_channel.channel_phase, index_list_to_remove)

        if channel_frequency_cut_max is not None:
            index_list_to_remove = []
            for count, value in enumerate(modified_channel.channel_frequency):
                if value > channel_frequency_cut_max:
                    index_list_to_remove.append(count)
            modified_channel.channel_frequency = np.delete(modified_channel.channel_frequency, index_list_to_remove)
            modified_channel.channel_impedance = np.delete(modified_channel.channel_impedance, index_list_to_remove)
            modified_channel.channel_phase = np.delete(modified_channel.channel_phase, index_list_to_remove)

        return modified_channel

    @staticmethod
    def copy(channel: Impedance) -> Impedance:
        """
        Create a deepcopy of Channel.

        :param channel: Impedance object
        :type channel: Impedance
        :return: Deepcopy of the impedance object
        :rtype: Impedance
        """
        return copy.deepcopy(channel)

    @staticmethod
    def from_waynekerr(csv_filename: str, channel_label: Optional[str] = None) -> 'Impedance':
        """
        Bring csv-data from wayne kerr 6515b to Impedance.

        :param csv_filename: .csv filename from impedance analyzer
        :type csv_filename: str
        :param channel_label: label to add to the Channel-class, optional.
        :type channel_label: str
        :return: Impedance object
        :rtype: HandleImpedance
        """
        impedance_measurement = np.genfromtxt(csv_filename, delimiter=',', dtype=float, skip_header=1,
                                              encoding='latin1')
        # move data to variables
        frequency = impedance_measurement[:, 0]
        impedance = impedance_measurement[:, 1]
        phase = impedance_measurement[:, 2]

        return HandleImpedance.generate_impedance_object(channel_frequency=frequency, channel_impedance=impedance, channel_phase=phase,
                                                         channel_source='Impedance Analyzer Wayne Kerr 6515b', channel_label=channel_label)

    @staticmethod
    def from_kemet_ksim(csv_filename: str) -> 'Impedance':
        """
        Import data from kemet "ksim" tool.

        :param csv_filename: path to csv-file
        :type csv_filename: str
        :return: Impedance object
        :rtype: Impedance
        """
        impedance_measurement = np.genfromtxt(csv_filename, delimiter=',', dtype=float, skip_header=1,
                                              encoding='latin1')
        # move data to variables
        frequency = impedance_measurement[:, 0]
        impedance = impedance_measurement[:, 3]
        phase = impedance_measurement[:, 4]

        return HandleImpedance.generate_impedance_object(channel_frequency=frequency, channel_impedance=impedance, channel_phase=phase,
                                                         channel_source='https://ksim3.kemet.com/capacitor-simulation')

    @staticmethod
    def plot_impedance(channel_list: List, figure_size: Tuple = None) -> None:
        """
        Plot and compare impedance channels.

        :param channel_list: List with impedances
        :type channel_list: List
        :param figure_size: figure size as tuple in inch, e.g. (4,3)
        :type figure_size: Tuple
        :return: None
        :rtype: None
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[x / 25.4 for x in figure_size] if figure_size is not None else None)
        for channel in channel_list:
            ax1.loglog(channel.channel_frequency, channel.channel_impedance, label=channel.channel_label,
                       color=channel.channel_color, linestyle=channel.channel_linestyle)
            ax2.semilogx(channel.channel_frequency, channel.channel_phase, label=channel.channel_label,
                         color=channel.channel_color, linestyle=channel.channel_linestyle)

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
        :return: None
        :rtype: None
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        for channel in channel_list:
            ax1.semilogx(channel.channel_frequency,
                         1e6 * channel.channel_impedance * np.sin(np.deg2rad(channel.channel_phase)) / channel.channel_frequency / 2 / np.pi,
                         label=channel.channel_label, color=channel.channel_color, linestyle=channel.channel_linestyle)
            ax2.semilogx(channel.channel_frequency,
                         channel.channel_impedance * np.cos(np.deg2rad(channel.channel_phase)),
                         label=channel.channel_label, color=channel.channel_color, linestyle=channel.channel_linestyle)

        ax1.grid()
        ax1.legend()
        ax1.set(xlabel='Frequency in Hz', ylabel=r'Inductance in $\rm{\mu}$H')

        ax2.grid()
        ax2.set(xlabel='Frequency in Hz', ylabel=r'AC Resistance in $\Omega$')
        ax2.legend()
        plt.show()

    @staticmethod
    def calc_re_im_parts(channel: Impedance, show_figure: bool = True):
        """
        Calculate real and imaginary part of Impedance measurement.

        :param channel: Impedance object
        :type channel: Impedance
        :param show_figure: Plot figure if true
        :type show_figure: bool
        :return: List with [(channel_frequency, frequency_real_part), (channel_frequency, frequency_imag_part)]
        :rtype: List
        """
        frequency_real_part = []
        frequency_imag_part = []
        complex_impedance = []

        for count_frequency, _ in enumerate(channel.channel_frequency):
            impedance = channel.channel_impedance[count_frequency] * np.exp(1j * channel.channel_phase[count_frequency] * 2 * np.pi / 360)
            complex_impedance.append(impedance)
            frequency_real_part.append(np.real(complex_impedance[count_frequency]))
            frequency_imag_part.append(np.imag(complex_impedance[count_frequency]))

        if show_figure:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax1.loglog(channel.channel_frequency, frequency_real_part)
            ax1.grid()
            ax1.set_xlabel('frequency in Hz')
            ax1.set_ylabel('real (impedance)')

            ax2.loglog(channel.channel_frequency, frequency_imag_part)
            ax2.set_xlabel('frequency in Hz')
            ax2.set_ylabel('imag (impedance)')
            ax2.grid()
            fig.show()

        frequency = copy.deepcopy(channel.channel_frequency)

        return [(frequency, frequency_real_part), (frequency, frequency_imag_part)]

    @staticmethod
    def calc_rlc(channel: Impedance, type_rlc: str, f_calc_c: float, f_calc_l: float, plot_figure: bool = False) -> tuple:
        """
        Calculate R, L, C values for given impedance curve.

        Calculated values will be drawn in a plot for comparison with the given data.
        :param channel: Impedance channel object
        :type channel: Impedance
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

        Example:
        >>> import pysignalscope as pss
        >>> example_data_rlc = pss.HandleImpedance.from_rlc('l', 1000, 500e-6, 10e-12)
        >>> recalculated_r, recalculated_l, recalculated_c = pss.HandleImpedance.calc_rlc(example_data_rlc, 'l', f_calc_l=10e3, f_calc_c=10e7, plot_figure=True)
        """
        # # Calculate R, L, C
        z_calc_c = np.interp(f_calc_c, channel.channel_frequency, channel.channel_impedance)
        z_calc_l = np.interp(f_calc_l, channel.channel_frequency, channel.channel_impedance)
        phase_calc_c = np.interp(f_calc_c, channel.channel_frequency, channel.channel_phase)
        phase_calc_l = np.interp(f_calc_l, channel.channel_frequency, channel.channel_phase)

        c_calc = 1 / (2 * np.pi * f_calc_c * z_calc_c)
        l_calc = z_calc_l / (2 * np.pi * f_calc_l)

        # # Calculate R at resonance frequency
        f_calc_r = 1 / (2 * np.pi * np.sqrt(l_calc * c_calc))
        z_calc_r = np.interp(f_calc_r, channel.channel_frequency, channel.channel_impedance)
        phase_calc_r = np.interp(f_calc_r, channel.channel_frequency, channel.channel_phase)
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
            recalculated_curve = HandleImpedance.from_rlc(type_rlc, r_calc, l_calc, c_calc)

            # generate plot
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

            # subplot 1 for impedance
            ax1.loglog(channel.channel_frequency, channel.channel_impedance, linestyle=linestyle_measurement, color=color_measurement, label='measurement')
            ax1.loglog(recalculated_curve.channel_frequency, abs(recalculated_curve.channel_impedance), linestyle=linestyle_calculation,
                       color=color_calculation, label='recalculated data')
            ax1.grid()
            ax1.legend()
            ax1.set(xlabel='Frequency in Hz', ylabel=r'Impedance in $\Omega$')
            ax1.plot(f_calc_c, z_calc_c, marker=markerstyle, color=color_measurement)
            ax1.plot(f_calc_r, z_calc_r, marker=markerstyle, color=color_measurement)
            ax1.plot(f_calc_l, z_calc_l, marker=markerstyle, color=color_measurement)

            # subplot 2 for phase
            ax2.semilogx(channel.channel_frequency, channel.channel_phase, linestyle=linestyle_measurement, color=color_measurement, label='measurement')
            ax2.semilogx(recalculated_curve.channel_frequency, recalculated_curve.channel_phase, linestyle=linestyle_calculation, color=color_calculation,
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
    def from_rlc(type_rlc: str, resistance: Union[float, np.array], inductance: Union[float, np.array], capacitance: Union[float, np.array]) -> 'Impedance':
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
        :rtype: Impedance

        Example:
        >>> import pysignalscope as pss
        >>> impedance_channel_object = pss.HandleImpedance.from_rlc('C', 10e-3, 100e-9, 36e-3)

         *  Type C and RLC
        ---R---L---C---

         *  Type R
        ---+---R---L---+---
           |           |
           +-----C-----+

         *  Type L
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

        return HandleImpedance.generate_impedance_object(channel_frequency=f, channel_impedance=np.abs(z_total), channel_phase=phase_vector)

    @staticmethod
    def check_capacitor_from_waynekerr(csv_filename: str, channel_label: str,
                                       target_capacitance: float, plot_figure: bool = True) -> 'Impedance':
        """
        Check a capacitor impedance .csv-curve against a target capacitance.

        Reads the .csv-curve from wayne kerr impedance analyzer, calculates r, l, and c from the measurement
        and shows the derivation from the target capacitance. Visualizes the curves in case of plot_figure is set
        to true.

        :param csv_filename: filepath to Wayne Kerr impedance analyzer .csv-file
        :type csv_filename: str
        :param channel_label: channel label for the plot
        :type channel_label: str
        :param target_capacitance: target capacitance in F
        :type target_capacitance: float
        :param plot_figure: Set to True for plot
        :type plot_figure: bool

        :return: measured capacitor as impedance curve
        :rtype: Impedance
        """
        f_calc_c = 100
        f_calc_l = 15e6

        capacitor_impedance = HandleImpedance.from_waynekerr(csv_filename, channel_label)

        r_measure, l_measure, c_measure = HandleImpedance.calc_rlc(capacitor_impedance, "c", f_calc_c, f_calc_l, plot_figure=plot_figure)
        derivation_capacitance = c_measure / target_capacitance

        print(f"{derivation_capacitance=}")

        return capacitor_impedance

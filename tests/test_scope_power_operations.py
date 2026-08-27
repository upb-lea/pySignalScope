"""Unit tests for power calculations and channel timebase validation."""

# python libraries
import os

# 3rd party libraries
import numpy as np
import numpy.testing
import pytest

# key must be set before import of pysignalscope. Disables the GUI inside the test machine.
os.environ["IS_TEST"] = "True"

# own libraries
import pysignalscope as pss
from pysignalscope.scope_dataclass import Channel


#########################################################################################################
# helper methods for test data
#########################################################################################################


def _channel(time, data, label=None, unit=None):
    """Generate a valid channel for the unit tests."""
    return pss.Scope.generate_channel(
        time=np.asarray(time, dtype=float),
        data=np.asarray(data, dtype=float),
        label=label,
        unit=unit,
    )


def _direct_channel(time, data):
    """Generate a Channel directly to allow invalid time/data shapes for unit tests."""
    return Channel(
        time=np.asarray(time, dtype=float),
        data=np.asarray(data, dtype=float),
        label=None,
        unit=None,
        color=None,
        linestyle=None,
        source=None,
        modulename="scope",
    )


#########################################################################################################
# test of _validate_compatible_timebases
#########################################################################################################


# parameterset for valid and invalid channel combinations
@pytest.mark.parametrize(
    "channels,valid_input_flag,exp_error,error_message",
    [
        # --valid inputs----
        # two channels with equal time data points
        ((_channel([0, 1, 2], [1, 2, 3]), _channel([0, 1, 2], [4, 5, 6])), True, None, None),
        # one channel is valid for methods which only need the shape check, e.g. integrate()
        ((_channel([0, 1], [1, 2]),), True, None, None),
        # --invalid inputs----
        # no channel input
        ((), False, ValueError, "Minimum one channel"),
        # wrong channel type
        ((1,), False, TypeError, "channel must be type Channel"),
        # time and data are not one-dimensional
        ((_direct_channel([[0, 1], [2, 3]], [[1, 2], [3, 4]]),), False, ValueError, "one-dimensional"),
        # time and data vectors have different lengths
        ((_direct_channel([0, 1, 2], [1, 2]),), False, ValueError, "must have the same shape"),
        # channel time vectors have different lengths
        ((_channel([0, 1, 2], [1, 2, 3]), _channel([0, 1], [4, 5])),
         False, ValueError, "time bases must have the same shape"),
        # channel time vectors have equal length but different time data points
        ((_channel([0, 1, 2], [1, 2, 3]), _channel([0, 1.1, 2], [4, 5, 6])),
         False, ValueError, "time bases must contain identical values"),
    ],
)
# definition of the testfunction
def test_validate_compatible_timebases(channels, valid_input_flag: bool, exp_error, error_message):
    """Test _validate_compatible_timebases() with valid and invalid channel inputs.

    :param channels: input channels for the protected validation method
    :type channels: tuple
    :param valid_input_flag: flag to indicate if valid input data is expected
    :type valid_input_flag: bool
    :param exp_error: expected error type for invalid input data
    :type exp_error: any
    :param error_message: part of the expected error message
    :type error_message: str or None
    """
    # Check if expected test result is no error
    if valid_input_flag is True:
        pss.Scope._validate_compatible_timebases(*channels)
    else:  # _validate_compatible_timebases raises an error
        with pytest.raises(exp_error, match=error_message):
            pss.Scope._validate_compatible_timebases(*channels)


#########################################################################################################
# test of multiply
#########################################################################################################


def test_multiply():
    """Test multiply() with valid voltage and current channels."""
    # Define input channels
    voltage = _channel([0, 1, 2], [2, 3, 4], label="Voltage", unit="V")
    current = _channel([0, 1, 2], [5, 6, 7], label="Current", unit="A")

    # calculate power from voltage and current
    power = pss.Scope.multiply(voltage, current)

    # verification of function result
    numpy.testing.assert_allclose(power.data, [10, 18, 28])
    numpy.testing.assert_array_equal(power.time, voltage.time)
    assert power.label == "Voltage * Current"
    assert power.unit == "W"


def test_multiply_calls_timebase_validation():
    """Test that multiply() uses _validate_compatible_timebases()."""
    # Define channels with different time data points
    voltage = _channel([0, 1, 2], [2, 3, 4])
    current = _channel([0, 1.1, 2], [5, 6, 7])

    # different time data points must raise a value error
    with pytest.raises(ValueError, match="time bases must contain identical values"):
        pss.Scope.multiply(voltage, current)


#########################################################################################################
# test of integrate
#########################################################################################################


def test_integrate_non_equidistant_time_steps():
    """Test integrate() with non-equidistant time data points."""
    # Define a power channel with two different timestep lengths
    power = _channel([0, 0.5, 2], [0, 2, 2], unit="W")

    # calculate cumulative energy
    energy = pss.Scope.integrate(power)

    # verification of function result
    numpy.testing.assert_allclose(energy.data, [0, 0.5, 3.5])
    assert energy.label == "Energy"
    assert energy.unit == "J"


def test_integrate_two_data_points():
    """Test integrate() with the minimum valid amount of two data points."""
    # Define a channel containing one valid integration interval
    power = _channel([0, 2], [3, 3], unit="W")

    # calculate cumulative energy with a user-defined label
    energy = pss.Scope.integrate(power, label="Turn-on energy")

    # verification of function result
    numpy.testing.assert_allclose(energy.data, [0, 6])
    assert energy.label == "Turn-on energy"


def test_integrate_calls_timebase_validation():
    """Test that integrate() uses _validate_compatible_timebases()."""
    # Generate an invalid channel with different time and data vector lengths
    power = _direct_channel([0, 1, 2], [1, 2])

    # invalid channel data shape must be detected by the validation method
    with pytest.raises(ValueError, match="must have the same shape"):
        pss.Scope.integrate(power)

"""Unit tests for the scope module."""

# python libraries
import pytest

# 3rd party libraries
import numpy as np
import numpy.testing

global IS_TEST
IS_TEST = True

# own libraries
import pysignalscope as pss



def test_generate_scope_object():
    """Test generate_scope_object() method."""
    # no input or missing input channel_time and channel_data - raise type error
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object()
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3])
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object(channel_data=[1, 2, 3])
    # different length of time and data must raise value error
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, 2])
    # invalid time data positive values
    with pytest.raises(ValueError):
        scope_object = pss.HandleScope.generate_scope_object(channel_time=[3, 2, 1], channel_data=[1, 2, 3])
    # invalid time data negative values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[-1, -2, -3], channel_data=[1, 2, 3])
    # invalid time data negative and positive values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[-1, -3, 1], channel_data=[1, 2, 3])

    # channel_time: non-equidistant values and negative valid values
    scope_object = pss.HandleScope.generate_scope_object(channel_time=[-3.3, -2.2, -1.1, 0, 1.2], channel_data=[-1, -2.1, -3.2, 4.4, -2.7])
    numpy.testing.assert_equal(scope_object.channel_time, [-3.3, -2.2, -1.1, 0, 1.2])
    numpy.testing.assert_equal(scope_object.channel_data, [-1, -2.1, -3.2, 4.4, -2.7])

    # channel_time: same x-data, should fail.
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3, 3, 4, 5], channel_data=[-1, -2.1, -3.2, 4.4])

    # valid positive data, mixed int and float
    scope_object = pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, 2, 3.1])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [1, 2, 3.1])

    # valid negative data, mixed int and float
    scope_object = pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[-1, -2.1, -3.2])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [-1, -2.1, -3.2])

    # valid mixed positive and negative data, mixed int and float
    scope_object = pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [1, -2.1, -3.2])

    # very high, very low and very small mixed values
    scope_object = pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1e25, -3.4e34, 3.1e-17])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [1e25, -3.4e34, 3.1e-17])

    # invalid time value
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[np.nan, 2, 3], channel_data=[0, 2, -3.2])

    # invalid data value
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[-np.nan, 2, -3.2])
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[-np.inf, 2, -3.2])
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[np.inf, 2, -3.2])

    # check None inputs
    scope_object = pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2])
    assert scope_object.channel_label is None
    assert scope_object.channel_unit is None
    assert scope_object.channel_color is None
    assert scope_object.channel_source is None
    assert scope_object.channel_linestyle is None

    # check inputs
    scope_object = pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2],
                                                         channel_label="test 1", channel_unit="A", channel_color="red",
                                                         channel_source="scope 11", channel_linestyle="--")
    assert scope_object.channel_label == "test 1"
    assert scope_object.channel_unit == "A"
    assert scope_object.channel_color == "red"
    assert scope_object.channel_source == "scope 11"
    assert scope_object.channel_linestyle == "--"

    # wrong type inputs
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_label=100.1)
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_unit=100.1)
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_color=100.1)
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_source=100.1)
    with pytest.raises(TypeError):
        pss.HandleScope.generate_scope_object(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_linestyle=100.1)

def test_from_numpy():
    """Test for the method from_numpy()."""
    time = [0, 1, 2 * np.pi]
    data = [2, 3, 2]
    frequency = 20000

    period_vector_t_i = np.array([time, data])

    # mode time
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="time")
    np.testing.assert_array_equal(scope_object.channel_time, time)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # no mode input, should behave same as radiant mode
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 2 / np.pi / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # mode radiant
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="rad", f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 2 / np.pi / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # mode degree
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="deg", f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 360 / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # Check for non set labels
    assert scope_object.channel_label is None
    assert scope_object.channel_unit is None

    # wrong input value for mode
    with pytest.raises(ValueError):
        pss.HandleScope.from_numpy(period_vector_t_i, "wrong_mode")

    with pytest.raises(ValueError):
        pss.HandleScope.from_numpy(period_vector_t_i, 100)

    # wrong input data to see if generate_scope_object() is used
    with pytest.raises(ValueError):
        pss.HandleScope.generate_scope_object(channel_time=[-3.14, -4, 5.0], channel_data=[1, -2.1, -3.2], channel_linestyle=100.1)

    # set label
    label = "trail_label"
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_label=label)
    assert scope_object.channel_label == label

    # set wrong label type
    with pytest.raises(TypeError):
        pss.HandleScope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_label=100)

    # set unit
    unit = "trial_unit"
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_unit=unit)
    assert scope_object.channel_unit == unit

    # set wrong unit type
    with pytest.raises(TypeError):
        pss.HandleScope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_unit=100)

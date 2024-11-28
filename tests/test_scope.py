"""Unit tests for the scope module."""

# python libraries
import pytest

# 3rd party libraries
import numpy as np

# own libraries
import pysignalscope as pss

def test_from_numpy():
    """Test for the function from_numpy()."""
    time = [0, 1, 2 * np.pi]
    data = [2, 3, 2]
    frequency = 20000

    period_vector_t_i = np.array([time, data])

    # mode time
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="time")
    np.testing.assert_array_equal(scope_object.channel_time, time)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # mode radiant
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="rad", f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 2 / np.pi / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # mode degree
    scope_object = pss.HandleScope.from_numpy(period_vector_t_i, mode="deg", f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 360 / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # wrong input value for mode
    with pytest.raises(ValueError):
        pss.HandleScope.from_numpy(period_vector_t_i, "wrong_mode")

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

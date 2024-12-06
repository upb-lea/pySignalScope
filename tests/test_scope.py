"""Unit tests for the scope module."""

# python libraries
import pytest
import os

# 3rd party libraries
import numpy as np
import numpy.testing

# key must be set before import of pysignalscope. Disables the GUI inside the test machine.
os.environ["IS_TEST"] = "True"

# own libraries
import pysignalscope as pss

def test_generate_scope_object():
    """Test generate_scope_object() method."""
    # no input or missing input channel_time and channel_data - raise type error
    with pytest.raises(TypeError):
        pss.Scope.generate_channel()
    with pytest.raises(TypeError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3])
    with pytest.raises(TypeError):
        pss.Scope.generate_channel(channel_data=[1, 2, 3])
    # different length of time and data must raise value error
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, 2])
    # invalid time data positive values
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[3, 2, 1], channel_data=[1, 2, 3])
    # invalid time data negative values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[-1, -2, -3], channel_data=[1, 2, 3])
    # invalid time data negative and positive values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[-1, -3, 1], channel_data=[1, 2, 3])

    # channel_time: non-equidistant values and negative valid values
    scope_object = pss.Scope.generate_channel(channel_time=[-3.3, -2.2, -1.1, 0, 1.2], channel_data=[-1, -2.1, -3.2, 4.4, -2.7])
    numpy.testing.assert_equal(scope_object.channel_time, [-3.3, -2.2, -1.1, 0, 1.2])
    numpy.testing.assert_equal(scope_object.channel_data, [-1, -2.1, -3.2, 4.4, -2.7])

    # channel_time: same x-data, should fail.
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3, 3, 4, 5], channel_data=[-1, -2.1, -3.2, 4.4])

    # valid positive data, mixed int and float
    scope_object = pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, 2, 3.1])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [1, 2, 3.1])

    # valid negative data, mixed int and float
    scope_object = pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[-1, -2.1, -3.2])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [-1, -2.1, -3.2])

    # valid mixed positive and negative data, mixed int and float
    scope_object = pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [1, -2.1, -3.2])

    # very high, very low and very small mixed values
    scope_object = pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1e25, -3.4e34, 3.1e-17])
    numpy.testing.assert_equal(scope_object.channel_time, [1, 2, 3])
    numpy.testing.assert_equal(scope_object.channel_data, [1e25, -3.4e34, 3.1e-17])

    # invalid time value
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[np.nan, 2, 3], channel_data=[0, 2, -3.2])

    # invalid data value
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[-np.nan, 2, -3.2])
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[-np.inf, 2, -3.2])
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[np.inf, 2, -3.2])

    # check None inputs
    scope_object = pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2])
    assert scope_object.channel_label is None
    assert scope_object.channel_unit is None
    assert scope_object.channel_color is None
    assert scope_object.channel_source is None
    assert scope_object.channel_linestyle is None

    # check inputs
    scope_object = pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2],
                                              channel_label="test 1", channel_unit="A", channel_color="red",
                                              channel_source="scope 11", channel_linestyle="--")
    assert scope_object.channel_label == "test 1"
    assert scope_object.channel_unit == "A"
    assert scope_object.channel_color == "red"
    assert scope_object.channel_source == "scope 11"
    assert scope_object.channel_linestyle == "--"

    # wrong type inputs
    with pytest.raises(TypeError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_label=100.1)
    with pytest.raises(TypeError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_unit=100.1)
    with pytest.raises(TypeError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_color=100.1)
    with pytest.raises(TypeError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_source=100.1)
    with pytest.raises(TypeError):
        pss.Scope.generate_channel(channel_time=[1, 2, 3], channel_data=[1, -2.1, -3.2], channel_linestyle=100.1)

def test_from_numpy():
    """Test for the method from_numpy()."""
    time = [0, 1, 2 * np.pi]
    data = [2, 3, 2]
    frequency = 20000

    period_vector_t_i = np.array([time, data])

    # mode time
    scope_object = pss.Scope.from_numpy(period_vector_t_i, mode="time")
    np.testing.assert_array_equal(scope_object.channel_time, time)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # no mode input, should behave same as radiant mode
    scope_object = pss.Scope.from_numpy(period_vector_t_i, f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 2 / np.pi / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # mode radiant
    scope_object = pss.Scope.from_numpy(period_vector_t_i, mode="rad", f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 2 / np.pi / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # mode degree
    scope_object = pss.Scope.from_numpy(period_vector_t_i, mode="deg", f0=frequency)
    np.testing.assert_array_almost_equal(scope_object.channel_time, np.array(time) / 360 / frequency)
    np.testing.assert_array_equal(scope_object.channel_data, data)

    # Check for non set labels
    assert scope_object.channel_label is None
    assert scope_object.channel_unit is None

    # wrong input value for mode
    with pytest.raises(ValueError):
        pss.Scope.from_numpy(period_vector_t_i, "wrong_mode")

    with pytest.raises(ValueError):
        pss.Scope.from_numpy(period_vector_t_i, 100)

    # wrong input data to see if generate_scope_object() is used
    with pytest.raises(ValueError):
        pss.Scope.generate_channel(channel_time=[-3.14, -4, 5.0], channel_data=[1, -2.1, -3.2], channel_linestyle=100.1)

    # set label
    label = "trail_label"
    scope_object = pss.Scope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_label=label)
    assert scope_object.channel_label == label

    # set wrong label type
    with pytest.raises(TypeError):
        pss.Scope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_label=100)

    # set unit
    unit = "trial_unit"
    scope_object = pss.Scope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_unit=unit)
    assert scope_object.channel_unit == unit

    # set wrong unit type
    with pytest.raises(TypeError):
        pss.Scope.from_numpy(period_vector_t_i, mode="rad", f0=frequency, channel_unit=100)

def test_low_pass_filter():
    """Unit test for low_pass_filter()."""
    # working test
    current_prim = pss.Scope.generate_channel([0, 1, 2, 3, 4, 5, 6], [1, 4, 2, 3, 7, 3, 2])
    filter_current_prim_1 = pss.Scope.low_pass_filter(current_prim, 1, angular_frequency_rad=0.3)
    numpy.testing.assert_array_almost_equal([0.99927604, 2.26610791, 2.85423117, 3.5885494, 4.09641649, 3.33691443, 1.99801723],
                                            filter_current_prim_1.channel_data)

    # working test for default values
    filter_current_prim_1 = pss.Scope.low_pass_filter(current_prim)
    numpy.testing.assert_array_almost_equal([0.7568143, 0.98001724, 1.15909406, 1.2985092, 1.37680805, 1.36477982, 1.29328745],
                                            filter_current_prim_1.channel_data)

    # insert not a scope type
    with pytest.raises(TypeError):
        pss.Scope.low_pass_filter(5, order=1, angular_frequency_rad=0.5)

    # wrong filter order type
    with pytest.raises(TypeError):
        pss.Scope.low_pass_filter(current_prim, order=1.4, angular_frequency_rad=0.5)
    # negative filter order
    with pytest.raises(ValueError):
        pss.Scope.low_pass_filter(current_prim, order=-3, angular_frequency_rad=0.5)

    # wrong filter frequency type
    with pytest.raises(TypeError):
        pss.Scope.low_pass_filter(current_prim, order=1, angular_frequency_rad=True)

    # wrong frequency value
    with pytest.raises(ValueError):
        pss.Scope.low_pass_filter(current_prim, order=1, angular_frequency_rad=1.4)
    with pytest.raises(ValueError):
        pss.Scope.low_pass_filter(current_prim, order=1, angular_frequency_rad=-2.2)

def test_derivative():
    """Test the derivative method."""
    # function test
    sample_scope_object = pss.Scope.generate_channel([0, 1, 2, 3, 4, 5, 6], [1, 4, 2, 3, 7, 3, 2])
    sample_scope_object_1st_derivative = pss.Scope.derivative(sample_scope_object, 1)
    numpy.testing.assert_equal([6, 1, 0, 2, 0, -2, 0], sample_scope_object_1st_derivative.channel_data)

    # function test using default order
    sample_scope_object_1st_derivative = pss.Scope.derivative(sample_scope_object)
    numpy.testing.assert_equal([6, 1, 0, 2, 0, -2, 0], sample_scope_object_1st_derivative.channel_data)

    # wrong scope type
    with pytest.raises(TypeError):
        pss.Scope.derivative(5, order=1)

    # wrong order type
    with pytest.raises(TypeError):
        pss.Scope.derivative(sample_scope_object, order=3.3)

    # negative oder type
    with pytest.raises(ValueError):
        pss.Scope.derivative(sample_scope_object, order=-2)

def test_eq():
    """Test __eq__()."""
    ch_1 = pss.Scope.generate_channel([1, 2, 3], [4, 5, 6],
                                      channel_unit="A", channel_label="label", channel_color="red",
                                      channel_source="source", channel_linestyle='--')
    ch_2 = pss.Scope.copy(ch_1)
    # assert both channels are the same
    assert ch_1 == ch_2

    # not the same: different time
    ch_2 = pss.Scope.copy(ch_1)
    ch_2.channel_time = np.array([2, 2.1, 3])
    assert not (ch_1 == ch_2)

    # not the same: different data
    ch_2 = pss.Scope.copy(ch_1)
    ch_2.channel_data = np.array([2, 2.1, 3])
    assert not (ch_1 == ch_2)

    # not the same: different units
    ch_2 = pss.Scope.copy(ch_1)
    ch_2 = pss.Scope.modify(ch_2, channel_unit="U")
    assert not (ch_1 == ch_2)

    # not the same: different labels
    ch_2 = pss.Scope.copy(ch_1)
    ch_2 = pss.Scope.modify(ch_2, channel_label="aaa")
    assert not (ch_1 == ch_2)

    # not the same: different colors
    ch_2 = pss.Scope.copy(ch_1)
    ch_2 = pss.Scope.modify(ch_2, channel_color="blue")
    assert not (ch_1 == ch_2)

    # not the same: different sources
    ch_2 = pss.Scope.copy(ch_1)
    ch_2 = pss.Scope.modify(ch_2, channel_source="asdf")
    assert not (ch_1 == ch_2)

    # not the same: different line styles
    ch_2 = pss.Scope.copy(ch_1)
    ch_2 = pss.Scope.modify(ch_2, channel_label=".-")
    assert not (ch_1 == ch_2)

def test_save_load():
    """Unit test for save and load."""
    # assumption: the given scope object is valid
    example = pss.Scope.generate_channel([1, 2, 3], [4, 5, 6],
                                         channel_unit="A", channel_label="label", channel_color="red",
                                         channel_source="source", channel_linestyle='--')

    # save + load: working example
    pss.Scope.save(example, "test_example")
    loaded_example = pss.Scope.load("test_example.pkl")
    assert example == loaded_example

    # save: wrong file path type
    with pytest.raises(TypeError):
        pss.Scope.save(example, 123)

    # save: insert wrong scope type
    with pytest.raises(TypeError):
        pss.Scope.save(123, "test_example")

    # load: not a pkl-file
    with pytest.raises(ValueError):
        pss.Scope.load("test_example.m")

    # load: not a string filepath
    with pytest.raises(TypeError):
        pss.Scope.load(123)

    # load: non-existing pkl-file
    with pytest.raises(ValueError):
        pss.Scope.load("test_example_not_existing.pkl")

def test_copy():
    """Unit test for copy()."""
    # wrong input type
    with pytest.raises(TypeError):
        pss.Scope.copy("not-a-channel")

    # test for valid copy
    object_to_copy = pss.Scope.generate_channel([1, 2, 3], [4, 5, 6],
                                                channel_unit="A", channel_label="label", channel_color="red",
                                                channel_source="source", channel_linestyle='--')
    object_copy = pss.Scope.copy(object_to_copy)

    assert object_to_copy == object_copy

def test_add():
    """Unit test for add()."""
    # sample data
    channel_1 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2], channel_data=[1, 2, 3])
    channel_2 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2], channel_data=[1, 2, 3])
    channel_3 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2], channel_data=[1, 2, 3])
    channel_4 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2], channel_data=[2, 4, 6])
    channel_5 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2, 3.3], channel_data=[2, 4, 6, 9])

    # wrong input type
    with pytest.raises(TypeError):
        pss.Scope.add(123, "wrong-type")
    # error: different length of channels
    with pytest.raises(ValueError):
        pss.Scope.add(channel_1, channel_5)
    # error: different time steps of channels
    with pytest.raises(ValueError):
        pss.Scope.add(channel_1, channel_2)

    # valid result
    channel_add = pss.Scope.add(channel_2, channel_3)
    np.testing.assert_equal(channel_add.channel_data, channel_4.channel_data)

def test_subtract():
    """Unit test for add()."""
    # sample data
    channel_1 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2], channel_data=[1, 2, 3])
    channel_2 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2], channel_data=[1, 2, 3])
    channel_3 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2], channel_data=[1, 2, 3])
    channel_4 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2], channel_data=[0, 0, 0])
    channel_5 = pss.Scope.generate_channel(channel_time=[1, 1.1, 2.2, 3.3], channel_data=[2, 4, 6, 9])

    # wrong input type
    with pytest.raises(TypeError):
        pss.Scope.subtract(123, "wrong-type")
    # error: different length of channels
    with pytest.raises(ValueError):
        pss.Scope.subtract(channel_1, channel_5)
    # error: different time steps of channels
    with pytest.raises(ValueError):
        pss.Scope.subtract(channel_1, channel_2)

    # valid result
    channel_subtract = pss.Scope.subtract(channel_2, channel_3)
    np.testing.assert_equal(channel_subtract.channel_data, channel_4.channel_data)

def test_mean():
    """Unit test for mean()."""
    # wrong input type
    with pytest.raises(TypeError):
        pss.Scope.calc_mean("not-a-scope-object")

    # mixed signal
    example_channel = pss.Scope.generate_channel([1, 2, 3, 4], [-1, 1, -1, 1])
    rms_example_channel = pss.Scope.calc_mean(example_channel)
    assert rms_example_channel == 0

    # positive signal, negative time steps
    example_channel = pss.Scope.generate_channel([-4.1, -3.1, -2.1, -1.1], [2.5, 1.5, 2.5, 1.5])
    rms_example_channel = pss.Scope.calc_mean(example_channel)
    assert rms_example_channel == 2

    # negative signal
    example_channel = pss.Scope.generate_channel([1, 2, 3, 4], [-2.5, -1.5, -2.5, -1.5])
    rms_example_channel = pss.Scope.calc_mean(example_channel)
    assert rms_example_channel == -2

    # non-equidistant time steps
    example_channel = pss.Scope.generate_channel([-2, -1, 1, 2], [-1, -1, 2, 2])
    rms_example_channel = pss.Scope.calc_mean(example_channel)
    assert rms_example_channel == 0.5

def test_abs_mean():
    """Unit test for abs_mean()."""
    # wrong input type
    with pytest.raises(TypeError):
        pss.Scope.calc_absmean("not-a-scope-object")

    # mixed signal
    example_channel = pss.Scope.generate_channel([1, 2, 3, 4], [-1, 1, -1, 1])
    rms_example_channel = pss.Scope.calc_absmean(example_channel)
    assert rms_example_channel == 1

    # positive signal, negative time steps
    example_channel = pss.Scope.generate_channel([-4.1, -3.1, -2.1, -1.1], [2.5, 1.5, 2.5, 1.5])
    rms_example_channel = pss.Scope.calc_absmean(example_channel)
    assert rms_example_channel == 2

    # negative signal
    example_channel = pss.Scope.generate_channel([1, 2, 3, 4], [-2.5, -1.5, -2.5, -1.5])
    rms_example_channel = pss.Scope.calc_absmean(example_channel)
    assert rms_example_channel == 2

    # non-equidistant time steps
    example_channel = pss.Scope.generate_channel([-2, -1, 1, 2], [-1, -1, 2, 2])
    rms_example_channel = pss.Scope.calc_absmean(example_channel)
    assert rms_example_channel == 1.5

def test_rms():
    """Unit test for rms()."""
    # wrong input type
    with pytest.raises(TypeError):
        pss.Scope.calc_rms("not-a-scope-object")

    # mixed signal
    example_channel = pss.Scope.generate_channel([1, 2, 3, 4], [-1, 1, -1, 1])
    rms_example_channel = pss.Scope.calc_rms(example_channel)
    assert rms_example_channel == 1

    # positive signal, negative time steps
    example_channel = pss.Scope.generate_channel([-4.1, -3.1, -2.1, -1.1], [2.5, 1.5, 2.5, 1.5])
    rms_example_channel = pss.Scope.calc_rms(example_channel)
    assert rms_example_channel == 2.0615528128088303

    # negative signal
    example_channel = pss.Scope.generate_channel([1, 2, 3, 4], [-2.5, -1.5, -2.5, -1.5])
    rms_example_channel = pss.Scope.calc_rms(example_channel)
    assert rms_example_channel == 2.0615528128088303

    # non-equidistant time steps
    example_channel = pss.Scope.generate_channel([-2, -1, 1, 2], [-1, -1, 2, 2])
    rms_example_channel = pss.Scope.calc_rms(example_channel)
    assert rms_example_channel == 1.5811388300841898

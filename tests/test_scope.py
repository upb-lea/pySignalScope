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

#########################################################################################################
# test of generate_channel
#########################################################################################################

# parameterset for values
@pytest.mark.parametrize("tst_channel_time,tst_channel_data,exp_scope_obj_or_err,error_flag", [
    # --invalid inputs----
    # no input or missing input channel_time and channel_data - raise type error
    (None, None, TypeError, True),
    ([1, 2, 3], None, TypeError, True),
    (None, [1, 2, 3], TypeError, True),
    # empty lists
    ([], [], ValueError, True),
    # different length of time and data must raise value error
    ([1, 2, 3], [1, 2], ValueError, True),
    # invalid time data positive values
    ([3, 2, 1], [1, 2, 3], ValueError, True),
    # invalid time data negative values. Time values in wrong order.
    ([-1, -2, -3], [1, 2, 3], ValueError, True),
    # invalid time data negative and positive values. Time values in wrong order.
    ([-1, -3, 1], [1, 2, 3], ValueError, True),
    # invalid time value
    ([np.nan, 2, 3], [0, 2, -3.2], ValueError, True),
    # invalid data value not defined, minus infinite, plus infinite
    ([1, 2, 3], [-np.nan, 2, -3.2], ValueError, True),
    ([1, 2, 3], [-np.inf, 2, -3.2], ValueError, True),
    ([1, 2, 3], [np.inf, 2, -3.2], ValueError, True),
    # --valid inputs----
    # channel_time: non-equidistant values and negative valid values
    ([-3.3, -2.2, -1.1, 0, 1.2], [-1, -2.1, -3.2, 4.4, -2.7], {"channel_time": [-3.3, -2.2, -1.1, 0, 1.2], "channel_data": [-1, -2.1, -3.2, 4.4, -2.7]}, False),
    # valid positive data, mixed int and float
    ([1, 2, 3], [1, 2, 3.1], {"channel_time": [1, 2, 3], "channel_data": [1, 2, 3.1]}, False),
    # valid negative data, mixed int and float
    ([1, 2, 3], [-1, -2.1, -3.2], {"channel_time": [1, 2, 3], "channel_data": [-1, -2.1, -3.2]}, False),
    # very high, very low and very small mixed values
    ([1, 2, 3], [1e25, -3.4e34, 3.1e-17], {"channel_time": [1, 2, 3], "channel_data": [1e25, -3.4e34, 3.1e-17]}, False)
])
# definition of the testfunction
def test_values_of_generate_channel(tst_channel_time: [], tst_channel_data: [], exp_scope_obj_or_err, error_flag: bool):
    """Test generate_channel() method according values.

    :param tst_channel_time: test time series
    :type tst_channel_time: Union[List[float], np.ndarray]
    :param tst_channel_data: test data series
    :type tst_channel_data: Union[List[float], np.ndarray]
    :param exp_scope_obj_or_err: expected channel time data and value data or expected error
    :type exp_scope_obj_or_err: any
    :param error_flag: flag to indicate, if an error or a valid result is expected
    :type error_flag: bool

    """
    # Check if expected test result is no error
    if error_flag is False:
        # channel_time: non-equidistant values and negative valid values
        scope_object = pss.Scope.generate_channel(channel_time=tst_channel_time, channel_data=tst_channel_data)
        numpy.testing.assert_equal(scope_object.time, exp_scope_obj_or_err["channel_time"])
        numpy.testing.assert_equal(scope_object.data, exp_scope_obj_or_err["channel_data"])
    else:  # generate_channel raises an error
        with pytest.raises(exp_scope_obj_or_err):
            pss.Scope.generate_channel(tst_channel_time, tst_channel_data)

# parameterset for attributes
@pytest.mark.parametrize(
    "tst_channel_label,tst_channel_unit,tst_channel_color,tst_channel_source,"
    "tst_channel_linestyle,exp_error, error_flag", [
        # --invalid inputs----
        # wrong type for one attribute
        (100.1, "A", None, "scope 11", None, TypeError, True),
        ("Label", 1, None, "scope 11", None, TypeError, True),
        ("Label", "Unit", 1.2, "scope 11", None, TypeError, True),
        ("Label", "Unit", "Color", 4, None, TypeError, True),
        ("Label", "Unit", "Color", "scope 11", 18, TypeError, True),
        # --valid inputs----
        # all inputs are set
        ("test 1", "A", "red", "scope 11", "--", None, False),
        # some input is set
        (None, "A", None, "scope 11", None, None, False),
        # no input is set
        (None, None, None, None, None, None, False)
    ])
# definition of the testfunction
def test_attributes_of_test_generate_channel(tst_channel_label, tst_channel_unit, tst_channel_color,
                                             tst_channel_source, tst_channel_linestyle, exp_error, error_flag: bool):
    """Test generate_channel() method according attributes.

    :param tst_channel_label: test variable for channel label
    :type tst_channel_label: any
    :param tst_channel_unit: test variable for channel unit
    :type tst_channel_unit: any
    :param tst_channel_color: test variable for channel color
    :type tst_channel_color: any
    :param tst_channel_source: test variable for channel source
    :type tst_channel_source: any
    :param tst_channel_linestyle: test variable for channel linestyle
    :type tst_channel_linestyle: any
    :param exp_error: expected error
    :type exp_error: any
    :param error_flag: flag to indicate, if an error or a valid result is expected
    :type error_flag: bool
    """
    # Define value input
    tst_channel_time = [-3.3, -2.2, -1.1, 0, 1.2]
    tst_channel_data = [-1, -2.1, -3.2, 4.4, -2.7]
    # Check if expected test result is no error
    if error_flag is False:
        # channel_time: non-equidistant values and negative valid values
        scope_object = pss.Scope.generate_channel(channel_time=tst_channel_time,
                                                  channel_data=tst_channel_data,
                                                  channel_label=tst_channel_label,
                                                  channel_unit=tst_channel_unit,
                                                  channel_color=tst_channel_color,
                                                  channel_source=tst_channel_source,
                                                  channel_linestyle=tst_channel_linestyle)
        # verification of function result
        numpy.testing.assert_equal(scope_object.time, tst_channel_time)
        numpy.testing.assert_equal(scope_object.data, tst_channel_data)
        numpy.testing.assert_equal(scope_object.label, tst_channel_label)
        numpy.testing.assert_equal(scope_object.unit, tst_channel_unit)
        numpy.testing.assert_equal(scope_object.color, tst_channel_color)
        numpy.testing.assert_equal(scope_object.source, tst_channel_source)
        numpy.testing.assert_equal(scope_object.linestyle, tst_channel_linestyle)
    else:  # generate_channel raises an error
        with pytest.raises(exp_error):
            scope_object = pss.Scope.generate_channel(channel_time=tst_channel_time,
                                                      channel_data=tst_channel_data,
                                                      channel_label=tst_channel_label,
                                                      channel_unit=tst_channel_unit,
                                                      channel_color=tst_channel_color,
                                                      channel_source=tst_channel_source,
                                                      channel_linestyle=tst_channel_linestyle)

#########################################################################################################
# test of from_numpy
#########################################################################################################


# test frequency
frequency1 = 200.6
frequency2 = 20000

# parameterset for values
@pytest.mark.parametrize("valid_input_flag, tst_mode, tst_f0, exp_factor_or_error, error_flag", [
    # --invalid inputs----
    # wrong type for one attribute
    (True, "wrong_mode", None, ValueError, True),
    (True, 2, None, ValueError, True),
    (True, None, "Freq", ValueError, True),
    (True, "rad", None, ValueError, True),
    (True, None, None, ValueError, True),
    (True, None, frequency1, ValueError, True),
    (False, "time", None, ValueError, True),
    # --valid inputs----
    (True, "time", None, 1, False),
    (True, "time", frequency1, 1, False),
    (True, "rad", frequency1, (1 / 2 / np.pi / frequency1), False),
    (True, "rad", frequency2, (1 / 2 / np.pi / frequency2), False),
    (True, "deg", frequency1, (1 / 360 / frequency1), False),
    (True, "deg", frequency2, (1 / 360 / frequency2), False)
])
# definition of the testfunction
def test_from_numpy(valid_input_flag: bool, tst_mode, tst_f0, exp_factor_or_error, error_flag: bool):
    """Test for the method from_numpy().

    :param valid_input_flag: flag to indicate that valid time and value data are to use
    :type valid_input_flag: bool
    :param tst_mode: test variable for mode
    :type tst_mode: any
    :param tst_f0: test variable for the frequency
    :type tst_f0: any
    :param exp_factor_or_error: expected factor for the result vector or expected error
    :type exp_factor_or_error: any
    :param error_flag: flag to indicate, if an error or a valid result is expected
    :type error_flag: bool
    """
    # Define value input
    tst_time = [1.1, 2.2, 3.3, 4]
    tst_time_invalid = [1.1, -1.1, 3.3, 4]
    tst_data = [-1, -2.1, -3.2, 4.4]

    tst_label = "DataLabel"
    tst_unit = "UnitOfData"

    # valid vector
    period_vector_t_i = np.array([tst_time, tst_data])
    # invalid vector
    period_vector_t_i_invalid = np.array([tst_time_invalid, tst_data])

    # Check if expected test result is no error
    if error_flag is False:
        # channel_time: non-equidistant values and negative valid values
        scope_object = pss.Scope.from_numpy(period_vector_t_i, mode=tst_mode, f0=tst_f0,
                                            channel_label=tst_label, channel_unit=tst_unit)
        # verification of attributes
        numpy.testing.assert_equal(scope_object.label, tst_label)
        numpy.testing.assert_equal(scope_object.unit, tst_unit)
        # verification of function result
        np.testing.assert_array_almost_equal(scope_object.time, np.array(tst_time) * exp_factor_or_error)
        np.testing.assert_array_equal(scope_object.data, tst_data)
    else:  # generate_channel raises an error
        if valid_input_flag is True:
            with pytest.raises(exp_factor_or_error):
                scope_object = pss.Scope.from_numpy(period_vector_t_i, mode=tst_mode, f0=tst_f0,
                                                    channel_label=tst_label, channel_unit=tst_unit)
        else:
            with pytest.raises(exp_factor_or_error):
                scope_object = pss.Scope.from_numpy(period_vector_t_i_invalid, mode=tst_mode, f0=tst_f0,
                                                    channel_label=tst_label, channel_unit=tst_unit)

#########################################################################################################
# test of low_pass_filter
#########################################################################################################


# channel values
test_vec1 = pss.Scope.generate_channel([0, 1, 2, 3, 4, 5, 6], [1, 4, 2, 3, 7, 3, 2])

# result values
test_res1 = [0.99927604, 2.26610791, 2.85423117, 3.5885494, 4.09641649, 3.33691443, 1.99801723]
test_res2 = [0.7568143, 0.98001724, 1.15909406, 1.2985092, 1.37680805, 1.36477982, 1.29328745]

# parameterset for values
@pytest.mark.parametrize("tst_vector, tst_order, tst_angular_freq, exp_result_or_error, error_flag", [
    # --invalid inputs----
    # wrong filter order type
    (test_vec1, 1.4, 0.5, TypeError, True),
    # negative filter order
    (test_vec1, -3, 0.5, ValueError, True),
    # wrong filter frequency type
    (test_vec1, 1, "Freq", TypeError, True),
    # wrong frequency value
    (test_vec1, 1, 1.4, ValueError, True),
    (test_vec1, 1, -2.2, ValueError, True),
    # insert not a scope type
    (5, 1, 0.3, TypeError, True),
    # --valid inputs----
    (test_vec1, 1, 0.3, test_res1, False),
    (test_vec1, None, None, test_res2, False),
])
# definition of the testfunction
def test_low_pass_filter(tst_vector, tst_order, tst_angular_freq, exp_result_or_error, error_flag: bool):
    """Unit test for low_pass_filter().

    :param tst_vector: test vector corresponds to scope object
    :type tst_vector: any
    :param tst_order: test variable for mode order or None indicates the default value test
    :type tst_order: any
    :param tst_angular_freq: test variable for the frequency angular_freq
    :type tst_angular_freq: any
    :param exp_result_or_error: expected test result vector or expected error
    :type exp_result_or_error: any
    :param error_flag: flag to indicate, if an error or a valid result is expected
    :type error_flag: bool
    """
    # Check if expected test result is no error
    if error_flag is False:
        # test default value test
        if isinstance(tst_order, (float, int)):
            # test with valid values
            filter_current_prim_1 = pss.Scope.low_pass_filter(tst_vector, tst_order, tst_angular_freq)
        else:
            # test of function default values
            filter_current_prim_1 = pss.Scope.low_pass_filter(tst_vector)

        # verification of function result
        numpy.testing.assert_array_almost_equal(exp_result_or_error, filter_current_prim_1.data)
    else:  # generate_channel raises an error
        with pytest.raises(exp_result_or_error):
            filter_current_prim_1 = pss.Scope.low_pass_filter(
                tst_vector, order=tst_order, angular_frequency_rad=tst_angular_freq)

#########################################################################################################
# test of derivative
#########################################################################################################


# channel values
test_vec1 = pss.Scope.generate_channel([0, 1, 2, 3, 4, 5, 6], [1, 4, 2, 3, 7, 3, 2])

# result values
test_res1 = [6, 1, 0, 2, 0, -2, 0]

# parameterset for values
@pytest.mark.parametrize("tst_vector, tst_order, exp_result_or_error, error_flag", [
    # --invalid inputs----
    # wrong order type
    (test_vec1, 3.3, TypeError, True),
    # negative order
    (test_vec1, -3, ValueError, True),
    # insert not a scope type
    (5, 1, TypeError, True),
    # --valid inputs----
    (test_vec1, 1, test_res1, False),
    (test_vec1, None, test_res1, False),
])
# definition of the testfunction
def test_derivative(tst_vector, tst_order, exp_result_or_error, error_flag: bool):
    """Test the derivative method.

    :param tst_vector: test vector corresponds to scope object
    :type tst_vector: any
    :param tst_order: test variable for mode order or None indicates the default value test
    :type tst_order: any
    :param exp_result_or_error: expected test result vector or expected error
    :type exp_result_or_error: any
    :param error_flag: flag to indicate, if an error or a valid result is expected
    :type error_flag: bool
    """
    # Check if expected test result is no error
    if error_flag is False:
        # test default value test
        if isinstance(tst_order, (float, int)):
            # test with valid values
            sample_scope_object_1st_derivative = pss.Scope.derivative(tst_vector, tst_order)
        else:
            # test of function default values
            sample_scope_object_1st_derivative = pss.Scope.derivative(tst_vector)

        # verification of function result
        numpy.testing.assert_array_equal(exp_result_or_error, sample_scope_object_1st_derivative.data)
    else:  # generate_channel raises an error
        with pytest.raises(exp_result_or_error):
            sample_scope_object_1st_derivative = pss.Scope.derivative(tst_vector, tst_order)

#########################################################################################################
# test of eq-operator (alternative stype)
#########################################################################################################

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
    ch_2.time = np.array([2, 2.1, 3])
    assert not (ch_1 == ch_2)

    # not the same: different data
    ch_2 = pss.Scope.copy(ch_1)
    ch_2.data = np.array([2, 2.1, 3])
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

#########################################################################################################
# test of save_load (alternative stype)
#########################################################################################################

def test_save_load():
    """Unit test for save and load."""
    # assumption: the given scope object is valid
    example = pss.Scope.generate_channel([-1, 2, 3.3], [-4, 5.0, 6.9],
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

#########################################################################################################
# test of copy (alternative stype)
#########################################################################################################

def test_copy():
    """Unit test for copy()."""
    # wrong input type
    with pytest.raises(TypeError):
        pss.Scope.copy("not-a-channel")

    # test for valid copy
    object_to_copy = pss.Scope.generate_channel([-1, 2, 3.3], [-4, 5.0, 6.9],
                                                channel_unit="A", channel_label="label", channel_color="red",
                                                channel_source="source", channel_linestyle='--')
    object_copy = pss.Scope.copy(object_to_copy)

    assert object_to_copy == object_copy

#########################################################################################################
# test of add (alternative stype)
#########################################################################################################

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
    np.testing.assert_equal(channel_add.data, channel_4.data)

#########################################################################################################
# test of subtract (alternative stype)
#########################################################################################################

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
    np.testing.assert_equal(channel_subtract.data, channel_4.data)

#########################################################################################################
# test of mean (alternative stype)
#########################################################################################################

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

#########################################################################################################
# test of abs_mean (alternative stype)
#########################################################################################################

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

#########################################################################################################
# test of rms (alternative stype)
#########################################################################################################

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

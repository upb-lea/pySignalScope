"""Unit tests for the impedance module."""

# python libraries
import pytest
import os

# 3rd party libraries
import numpy as np

# key must be set before import of pysignalscope. Disables the GUI inside the test machine.
os.environ["IS_TEST"] = "True"

# own libraries
import pysignalscope as pss


def test_generate_impedance_object():
    """Test generate_impedance_object() method."""
    # no input or missing input channel_frequency and channel_impedance - raise type error
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object()
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3])
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object(channel_impedance=[1, 2, 3])
    # different length of time and data must raise value error
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, 2], channel_phase=[1, 2, 3])
    # invalid time data positive values
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[3, 2, 1], channel_impedance=[1, 2, 3], channel_phase=[1, 2, 3])
    # invalid time data negative values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[-1, -2, -3], channel_impedance=[1, 2, 3], channel_phase=[1, 2, 3])
    # invalid time data negative and positive values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[-1, -3, 1], channel_impedance=[1, 2, 3], channel_phase=[1, 2, 3])

    # empty frequency, impedance and phase
    with (pytest.raises(ValueError)):
        pss.Impedance.generate_impedance_object(channel_frequency=[], channel_impedance=[1, 2, 3], channel_phase=[1, 2, 3])
    with (pytest.raises(ValueError)):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[], channel_phase=[1, 2, 3])
    with (pytest.raises(ValueError)):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, 2, 3], channel_phase=[])

    # channel_frequency: non-equidistant values and negative valid values
    impedance_object = pss.Impedance.generate_impedance_object(channel_frequency=[-3.3, -2.2, -1.1, 0, 1.2],
                                                               channel_impedance=[-1, -2.1, -3.2, 4.4, -2.7], channel_phase=[-1, -2.1, -4, 3.3, 5])
    np.testing.assert_equal(impedance_object.frequency, [-3.3, -2.2, -1.1, 0, 1.2])
    np.testing.assert_equal(impedance_object.impedance, [-1, -2.1, -3.2, 4.4, -2.7])
    np.testing.assert_equal(impedance_object.phase, [-1, -2.1, -4, 3.3, 5])

    # channel_frequency: same x-data, should fail.
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3, 3, 4, 5], channel_impedance=[-1, -2.1, -3.2, 4.4],
                                                channel_phase=[-1, -2.1, -4, 3.3, 5])

    # valid positive data, mixed int and float
    impedance_object = pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, 2, 3.1], channel_phase=[1, 2.1, 4])
    np.testing.assert_equal(impedance_object.frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.impedance, [1, 2, 3.1])
    np.testing.assert_equal(impedance_object.phase, [1, 2.1, 4])

    # valid negative data, mixed int and float
    impedance_object = pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[-1, -2.1, -3.2],
                                                               channel_phase=[-2, -2.2, -5])
    np.testing.assert_equal(impedance_object.frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.impedance, [-1, -2.1, -3.2])
    np.testing.assert_equal(impedance_object.phase, [-2, -2.2, -5])

    # valid mixed positive and negative data, mixed int and float
    impedance_object = pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                               channel_phase=[2, -3.3, -3])
    np.testing.assert_equal(impedance_object.frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.impedance, [1, -2.1, -3.2])
    np.testing.assert_equal(impedance_object.phase, [2, -3.3, -3])

    # very high, very low and very small mixed values
    impedance_object = pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1e25, -3.4e34, 3.1e-17],
                                                               channel_phase=[1e32, -3.3e33, 3.33e-17])
    np.testing.assert_equal(impedance_object.frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.impedance, [1e25, -3.4e34, 3.1e-17])
    np.testing.assert_equal(impedance_object.phase, [1e32, -3.3e33, 3.33e-17])

    # invalid frequency value
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[np.nan, 2, 3], channel_impedance=[0, 2, -3.2], channel_phase=[1, 2, 3.3])

    # invalid impedance value
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[-np.nan, 2, -3.2], channel_phase=[1, 2, 3.3])
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[-np.inf, 2, -3.2], channel_phase=[1, 2, 3.3])
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[np.inf, 2, -3.2], channel_phase=[1, 2, 3.3])

    # invalid phase value
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, 2, -3.2], channel_phase=[-np.nan, 2, 3.3])
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[2, 2, -3.2], channel_phase=[-np.inf, 2, 3.3])
    with pytest.raises(ValueError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[2, 2, -3.2], channel_phase=[np.inf, 2, 3.3])

    # check None inputs
    impedance_object = pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2], channel_phase=[1, 2, 3.3])
    assert impedance_object.label is None
    assert impedance_object.unit is None
    assert impedance_object.color is None
    assert impedance_object.source is None
    assert impedance_object.linestyle is None

    # check inputs
    impedance_object = pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2], channel_phase=[1, 2, 3.3],
                                                               channel_label="test 1", channel_unit="A", channel_color="red",
                                                               channel_source="scope 11", channel_linestyle="--")
    assert impedance_object.label == "test 1"
    assert impedance_object.unit == "A"
    assert impedance_object.color == "red"
    assert impedance_object.source == "scope 11"
    assert impedance_object.linestyle == "--"

    # wrong type inputs
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                channel_phase=[1, 2, 3.3], channel_label=100.1)
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                channel_phase=[1, 2, 3.3], channel_unit=100.1)
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                channel_phase=[1, 2, 3.3], channel_color=100.1)
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                channel_phase=[1, 2, 3.3], channel_source=100.1)
    with pytest.raises(TypeError):
        pss.Impedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                channel_phase=[1, 2, 3.3], channel_linestyle=100.1)


def test_save_load():
    """Unit test for save and load."""
    # assumption: the given scope object is valid
    example = pss.Impedance.generate_impedance_object([0.1, 1, 1.2, 2.1, 3], [0, 5.1, 1.2, 2.1, 3],
                                                      channel_phase=[-4, -5.9, 0, 6, 1.2],
                                                      channel_unit="A", channel_label="label", channel_color="red",
                                                      channel_source="source", channel_linestyle='--')

    # save + load: working example
    pss.Impedance.save(example, "test_example")
    loaded_example = pss.Impedance.load("test_example.pkl")
    assert (example.frequency == loaded_example.frequency).all()
    assert (example.impedance == loaded_example.impedance).all()
    assert (example.phase == loaded_example.phase).all()
    assert example.unit == loaded_example.unit
    assert example.label == loaded_example.label
    assert example.color == loaded_example.color
    assert example.source == loaded_example.source
    assert example.linestyle == loaded_example.linestyle

    # save: wrong file path type
    with pytest.raises(TypeError):
        pss.Impedance.save(example, 123)

    # save: insert wrong scope type
    with pytest.raises(TypeError):
        pss.Impedance.save(123, "test_example")

    # load: not a pkl-file
    with pytest.raises(ValueError):
        pss.Impedance.load("test_example.m")

    # load: not a string filepath
    with pytest.raises(TypeError):
        pss.Impedance.load(123)

    # load: non-existing pkl-file
    with pytest.raises(ValueError):
        pss.Impedance.load("test_example_not_existing.pkl")

def test_copy():
    """Unit test for copy()."""
    # wrong input type
    with pytest.raises(TypeError):
        pss.Impedance.copy("not-a-channel")

    # test for valid copy
    object_to_copy = pss.Impedance.generate_impedance_object([0.1, 1, 1.2, 2.1, 3], [0, 5.1, 1.2, 2.1, 3], [-4, -5.9, 0, 6, 1.2],
                                                             channel_unit="A", channel_label="label", channel_color="red",
                                                             channel_source="source", channel_linestyle='--')
    object_copy = pss.Impedance.copy(object_to_copy)

    assert object_copy == object_copy

def test_eq():
    """Test __eq__()."""
    ch_1 = pss.Impedance.generate_impedance_object([1, 2, 3], [4, 5, 6], [7, 8, 9],
                                                   channel_unit="A", channel_label="label", channel_color="red",
                                                   channel_source="source", channel_linestyle='--')
    ch_2 = pss.Impedance.copy(ch_1)
    # assert both channels are the same
    assert ch_1 == ch_2

    # not the same: different frequency
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2.frequency = np.array([2, 2.1, 3])
    assert not (ch_1 == ch_2)

    # not the same: different impedance data
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2.impedance = np.array([2, 2.1, 3])
    assert not (ch_1 == ch_2)

    # not the same: different phase
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2.phase = np.array([2, 2.1, 3])
    assert not (ch_1 == ch_2)

    # not the same: different units
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2 = pss.Impedance.modify(ch_2, channel_unit="U")
    assert not (ch_1 == ch_2)

    # not the same: different labels
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2 = pss.Impedance.modify(ch_2, channel_label="aaa")
    assert not (ch_1 == ch_2)

    # not the same: different colors
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2 = pss.Impedance.modify(ch_2, channel_color="blue")
    assert not (ch_1 == ch_2)

    # not the same: different sources
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2 = pss.Impedance.modify(ch_2, channel_source="asdf")
    assert not (ch_1 == ch_2)

    # not the same: different line styles
    ch_2 = pss.Impedance.copy(ch_1)
    ch_2 = pss.Impedance.modify(ch_2, channel_label=".-")
    assert not (ch_1 == ch_2)

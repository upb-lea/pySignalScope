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
        pss.HandleImpedance.generate_impedance_object()
    with pytest.raises(TypeError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3])
    with pytest.raises(TypeError):
        pss.HandleImpedance.generate_impedance_object(channel_impedance=[1, 2, 3])
    # different length of time and data must raise value error
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, 2], channel_phase=[1, 2, 3])
    # invalid time data positive values
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[3, 2, 1], channel_impedance=[1, 2, 3], channel_phase=[1, 2, 3])
    # invalid time data negative values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[-1, -2, -3], channel_impedance=[1, 2, 3], channel_phase=[1, 2, 3])
    # invalid time data negative and positive values. Time values in wrong order.
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[-1, -3, 1], channel_impedance=[1, 2, 3], channel_phase=[1, 2, 3])

    # channel_frequency: non-equidistant values and negative valid values
    impedance_object = pss.HandleImpedance.generate_impedance_object(channel_frequency=[-3.3, -2.2, -1.1, 0, 1.2],
                                                                     channel_impedance=[-1, -2.1, -3.2, 4.4, -2.7], channel_phase=[-1, -2.1, -4, 3.3, 5])
    np.testing.assert_equal(impedance_object.channel_frequency, [-3.3, -2.2, -1.1, 0, 1.2])
    np.testing.assert_equal(impedance_object.channel_impedance, [-1, -2.1, -3.2, 4.4, -2.7])
    np.testing.assert_equal(impedance_object.channel_phase, [-1, -2.1, -4, 3.3, 5])

    # channel_frequency: same x-data, should fail.
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3, 3, 4, 5], channel_impedance=[-1, -2.1, -3.2, 4.4],
                                                      channel_phase=[-1, -2.1, -4, 3.3, 5])

    # valid positive data, mixed int and float
    impedance_object = pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, 2, 3.1], channel_phase=[1, 2.1, 4])
    np.testing.assert_equal(impedance_object.channel_frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.channel_impedance, [1, 2, 3.1])
    np.testing.assert_equal(impedance_object.channel_phase, [1, 2.1, 4])

    # valid negative data, mixed int and float
    impedance_object = pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[-1, -2.1, -3.2],
                                                                     channel_phase=[-2, -2.2, -5])
    np.testing.assert_equal(impedance_object.channel_frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.channel_impedance, [-1, -2.1, -3.2])
    np.testing.assert_equal(impedance_object.channel_phase, [-2, -2.2, -5])

    # valid mixed positive and negative data, mixed int and float
    impedance_object = pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                                     channel_phase=[2, -3.3, -3])
    np.testing.assert_equal(impedance_object.channel_frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.channel_impedance, [1, -2.1, -3.2])
    np.testing.assert_equal(impedance_object.channel_phase, [2, -3.3, -3])

    # very high, very low and very small mixed values
    impedance_object = pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1e25, -3.4e34, 3.1e-17],
                                                                     channel_phase=[1e32, -3.3e33, 3.33e-17])
    np.testing.assert_equal(impedance_object.channel_frequency, [1, 2, 3])
    np.testing.assert_equal(impedance_object.channel_impedance, [1e25, -3.4e34, 3.1e-17])
    np.testing.assert_equal(impedance_object.channel_phase, [1e32, -3.3e33, 3.33e-17])

    # invalid frequency value
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[np.nan, 2, 3], channel_impedance=[0, 2, -3.2], channel_phase=[1, 2, 3.3])

    # invalid impedance value
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[-np.nan, 2, -3.2], channel_phase=[1, 2, 3.3])
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[-np.inf, 2, -3.2], channel_phase=[1, 2, 3.3])
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[np.inf, 2, -3.2], channel_phase=[1, 2, 3.3])

    # invalid phase value
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, 2, -3.2], channel_phase=[-np.nan, 2, 3.3])
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[2, 2, -3.2], channel_phase=[-np.inf, 2, 3.3])
    with pytest.raises(ValueError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[2, 2, -3.2], channel_phase=[np.inf, 2, 3.3])

    # check None inputs
    impedance_object = pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2], channel_phase=[1, 2, 3.3])
    assert impedance_object.channel_label is None
    assert impedance_object.channel_unit is None
    assert impedance_object.channel_color is None
    assert impedance_object.channel_source is None
    assert impedance_object.channel_linestyle is None

    # check inputs
    impedance_object = pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2], channel_phase=[1, 2, 3.3],
                                                                     channel_label="test 1", channel_unit="A", channel_color="red",
                                                                     channel_source="scope 11", channel_linestyle="--")
    assert impedance_object.channel_label == "test 1"
    assert impedance_object.channel_unit == "A"
    assert impedance_object.channel_color == "red"
    assert impedance_object.channel_source == "scope 11"
    assert impedance_object.channel_linestyle == "--"

    # wrong type inputs
    with pytest.raises(TypeError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                      channel_phase=[1, 2, 3.3], channel_label=100.1)
    with pytest.raises(TypeError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                      channel_phase=[1, 2, 3.3], channel_unit=100.1)
    with pytest.raises(TypeError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                      channel_phase=[1, 2, 3.3], channel_color=100.1)
    with pytest.raises(TypeError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                      channel_phase=[1, 2, 3.3], channel_source=100.1)
    with pytest.raises(TypeError):
        pss.HandleImpedance.generate_impedance_object(channel_frequency=[1, 2, 3], channel_impedance=[1, -2.1, -3.2],
                                                      channel_phase=[1, 2, 3.3], channel_linestyle=100.1)


def test_save_load():
    """Unit test for save and load."""
    # assumption: the given scope object is valid
    example = pss.HandleImpedance.generate_impedance_object([1, 2, 3], [4, 5, 6],
                                                            channel_phase=[10, 20, 30],
                                                            channel_unit="A", channel_label="label", channel_color="red",
                                                            channel_source="source", channel_linestyle='--')

    # save + load: working example
    pss.HandleImpedance.save(example, "test_example")
    loaded_example = pss.HandleImpedance.load("test_example.pkl")
    assert (example.channel_frequency == loaded_example.channel_frequency).all()
    assert (example.channel_impedance == loaded_example.channel_impedance).all()
    assert (example.channel_phase == loaded_example.channel_phase).all()
    assert example.channel_unit == loaded_example.channel_unit
    assert example.channel_label == loaded_example.channel_label
    assert example.channel_color == loaded_example.channel_color
    assert example.channel_source == loaded_example.channel_source
    assert example.channel_linestyle == loaded_example.channel_linestyle

    # save: wrong file path type
    with pytest.raises(TypeError):
        pss.HandleImpedance.save(example, 123)

    # save: insert wrong scope type
    with pytest.raises(TypeError):
        pss.HandleImpedance.save(123, "test_example")

    # load: not a pkl-file
    with pytest.raises(ValueError):
        pss.HandleImpedance.load("test_example.m")

    # load: not a string filepath
    with pytest.raises(TypeError):
        pss.HandleImpedance.load(123)

    # load: non-existing pkl-file
    with pytest.raises(ValueError):
        pss.HandleImpedance.load("test_example_not_existing.pkl")
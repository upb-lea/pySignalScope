"""Unit tests for the impedance module."""

# python libraries
import pytest
import os

# 3rd party libraries

# key must be set before import of pysignalscope. Disables the GUI inside the test machine.
os.environ["IS_TEST"] = "True"

# own libraries
import pysignalscope as pss

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

# -*- coding: utf-8 -*-
"""
Unit tests for components/nidaqmx_hardware_multitask.py's _create_channel,
using a mocked nidaqmx Task (no real NI hardware/driver required).

Covers the fix requested after a bug report: a Channel Table's Coupling,
Current Excitation Source, and Current Excitation Value entries must always
be applied, regardless of the channel's Type (Acceleration/Force/Voltage).
Previously, a "Voltage"-typed channel silently never received current
excitation at all (so an IEPE/ICP sensor wired to it was never powered), and
Coupling was never applied to any channel type.
"""
import sys
import os
from unittest.mock import MagicMock
import pytest
import nidaqmx.constants as nic


class FakeChannel:
    """A plain attribute bag standing in for the nidaqmx channel object.

    Deliberately not a MagicMock for this role: MagicMock only records
    method *calls* in mock_calls, not plain attribute assignments
    (``channel.ai_coupling = x``), which is exactly what _create_channel
    does -- so a MagicMock would make "was this attribute ever set" checks
    silently meaningless. A plain object lets tests use hasattr()/getattr()
    to check that directly.
    """
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.nidaqmx_hardware_multitask import NIDAQmxAcquisition, _parse_coupling
from components.utilities import Channel


def make_channel(channel_type, coupling, excitation_source, excitation, sensitivity=100, unit='g'):
    return Channel(
        node_number='1', node_direction='Z', comment='', serial_number='',
        triax_dof='', sensitivity=sensitivity, unit=unit, make='', model='', expiration='',
        physical_device='cDAQ1Mod1', physical_channel='ai0', channel_type=channel_type,
        minimum_value=-5, maximum_value=5, coupling=coupling,
        excitation_source=excitation_source, excitation=excitation,
        feedback_device=None, feedback_channel=None, warning_level=None, abort_level=None)


def make_acquisition_with_mock_task():
    acq = NIDAQmxAcquisition()
    acq.task = MagicMock()
    mock_channel = FakeChannel()
    acq.task.ai_channels.add_ai_voltage_chan.return_value = mock_channel
    acq.task.ai_channels.add_ai_accel_chan.return_value = mock_channel
    acq.task.ai_channels.add_ai_force_iepe_chan.return_value = mock_channel
    return acq, mock_channel


#%% Coupling parsing

def test_parse_coupling_accepts_standard_and_iepe_aliases():
    assert _parse_coupling('AC') == nic.Coupling.AC
    assert _parse_coupling('dc') == nic.Coupling.DC
    assert _parse_coupling('IEPE') == nic.Coupling.AC
    assert _parse_coupling('icp') == nic.Coupling.AC
    assert _parse_coupling('ccld') == nic.Coupling.AC
    assert _parse_coupling('GND') == nic.Coupling.GND


def test_parse_coupling_blank_or_none_means_leave_default():
    assert _parse_coupling(None) is None
    assert _parse_coupling('') is None
    assert _parse_coupling('   ') is None


def test_parse_coupling_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_coupling('not a real coupling')


#%% Voltage-type channel: excitation must now be applied

def test_voltage_channel_receives_current_excitation_when_requested():
    acq, mock_channel = make_acquisition_with_mock_task()
    channel_data = make_channel(channel_type='voltage', coupling='IEPE',
                                 excitation_source='internal', excitation=0.004)
    acq._create_channel(channel_data)

    acq.task.ai_channels.add_ai_voltage_chan.assert_called_once()
    assert mock_channel.ai_excit_src == nic.ExcitationSource.INTERNAL
    assert mock_channel.ai_excit_val == 0.004
    assert mock_channel.ai_excit_voltage_or_current == nic.ExcitationVoltageOrCurrent.USE_CURRENT
    assert mock_channel.ai_coupling == nic.Coupling.AC


def test_voltage_channel_no_excitation_when_source_is_none():
    acq, mock_channel = make_acquisition_with_mock_task()
    channel_data = make_channel(channel_type='voltage', coupling=None,
                                 excitation_source='none', excitation=0)
    acq._create_channel(channel_data)

    # ai_excit_src/ai_excit_val/ai_coupling must not have been touched
    assert not hasattr(mock_channel, 'ai_excit_src')
    assert not hasattr(mock_channel, 'ai_excit_val')
    assert not hasattr(mock_channel, 'ai_coupling')


#%% Coupling must now be applied to Acceleration and Force channels too

def test_acceleration_channel_receives_coupling():
    acq, mock_channel = make_acquisition_with_mock_task()
    channel_data = make_channel(channel_type='acceleration', coupling='AC',
                                 excitation_source='internal', excitation=0.004, unit='g')
    acq._create_channel(channel_data)

    acq.task.ai_channels.add_ai_accel_chan.assert_called_once()
    call_kwargs = acq.task.ai_channels.add_ai_accel_chan.call_args.kwargs
    assert call_kwargs['current_excit_source'] == nic.ExcitationSource.INTERNAL
    assert call_kwargs['current_excit_val'] == 0.004
    assert mock_channel.ai_coupling == nic.Coupling.AC


def test_force_channel_receives_coupling():
    acq, mock_channel = make_acquisition_with_mock_task()
    channel_data = make_channel(channel_type='force', coupling='DC',
                                 excitation_source='internal', excitation=0.004, unit='N')
    acq._create_channel(channel_data)

    acq.task.ai_channels.add_ai_force_iepe_chan.assert_called_once()
    assert mock_channel.ai_coupling == nic.Coupling.DC


def test_blank_coupling_leaves_device_default_untouched():
    acq, mock_channel = make_acquisition_with_mock_task()
    channel_data = make_channel(channel_type='voltage', coupling='',
                                 excitation_source='none', excitation=0)
    acq._create_channel(channel_data)
    assert not hasattr(mock_channel, 'ai_coupling')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

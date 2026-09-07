# -*- coding: utf-8 -*-
"""
Constant-Force Chirp environment: a single-shaker, single-force-control-
channel environment that sweeps a chirp while continuously trimming its
output voltage amplitude so the *measured force* (not the output voltage)
stays at a constant target level across the sweep. Accelerometers (and any
other acquired channels) are monitoring/abort channels only -- they are
never a target of control.

Scope note (read this before extending)
----------------------------------------
This module is new code, not a refactor of an existing Rattlesnake
environment. It reuses, unmodified:
  * ``AbstractSysIdMetadata`` / ``AbstractSysIdUI`` / ``AbstractSysIdEnvironment``
    (components/abstract_sysid_environment.py) for Phase 1 (system
    identification of H_FV(f) = F(f)/V(f) via Rattlesnake's existing H1/H2/H3/Hv
    estimator in components/spectral_processing.py, using the existing
    ``ChirpSignalGenerator`` as the identification excitation).
  * ``components/data_collector.py``, ``components/signal_generation_process.py``,
    ``components/spectral_processing.py`` process/queue machinery, unchanged.
It adds new code for Phase 2/3 (feedforward + closed-loop force trim):
  * ``control_laws/constant_force_chirp_control.py`` -- the control algorithm
    itself (framework-agnostic, unit tested, see tests/).
  * ``ConstantForceChirpSignalGenerator`` in components/signal_generation.py --
    a continuously phase-advancing chirp whose amplitude can be updated
    between frames (additive; does not change any existing signal generator).
  * ``ConstantForceChirpControlProcess`` below -- a new subprocess that plays
    the same structural role as ``RandomVibrationDataAnalysisProcess.run_control``
    (components/random_vibration_sys_id_data_analysis.py): it continuously
    drains newly acquired frames and pushes updated drive parameters back to
    signal generation. Unlike that class, it corrects a single scalar
    amplitude trim (informed by a synchronous/lock-in force-amplitude
    estimate) rather than solving a CPSD-matrix pseudoinverse each update.

IMPORTANT hardware wiring requirement
--------------------------------------
Rattlesnake's system-identification FRF estimator (H1/H2/H3/Hv) correlates
two *acquired* channels against each other -- it does not have access to the
raw digital command value. To identify H_FV(f) = F(f)/V(f), the "reference"
channel for this environment must therefore be an acquired voltage-monitor
channel (a loopback from the amplifier input, or the shaker input, wired
into a spare DAQ analog input and configured with that channel's
``feedback_device``/``feedback_channel`` fields -- see ``Channel`` in
components/utilities.py), not merely "the channel we happen to be driving".
Without this, H_FV(f) would be identified against the *commanded* signal and
silently absorb whatever gain/phase the amplifier and cabling add -- which is
exactly the error the feedforward inversion is meant to correct for.

Control-phase data collector windowing requirement
----------------------------------------------------
The closed-loop trim (Phase 3) reconstructs the acquired force/acceleration
time samples via inverse FFT of the frames the data collector forwards (see
``control_laws.constant_force_chirp_control.fft_bin_amplitude`` for why, and
why this is exact only for a rectangular window). The control-phase
``CollectorMetadata`` built in this module therefore always requests
``Window.RECTANGLE``. The identification-phase collector (used only during
Phase 1) is unaffected and continues to use whatever window
``AbstractSysIdMetadata.sysid_window`` specifies, exactly as the other
sys-id-based environments do.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from enum import Enum
import multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np
import netCDF4 as nc4
from qtpy import QtWidgets, uic
from qtpy.QtCore import Qt

from .abstract_sysid_environment import AbstractSysIdMetadata, AbstractSysIdUI, AbstractSysIdEnvironment
from .abstract_message_process import AbstractMessageProcess
from .environments import (ControlTypes, environment_definition_ui_paths,
                            environment_run_ui_paths)
from .utilities import VerboseMessageQueue, GlobalCommands, flush_queue
from .signal_generation import ConstantForceChirpSignalGenerator
from .signal_generation_process import (signal_generation_process, SignalGenerationCommands,
                                         SignalGenerationMetadata)
from .spectral_processing import spectral_processing_process
from .abstract_sysid_data_analysis import sysid_data_analysis_process, SysIDDataAnalysisCommands
from .data_collector import (data_collector_process, DataCollectorCommands, CollectorMetadata,
                              AcquisitionType, Acceptance, TriggerSlope, Window)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control_laws.constant_force_chirp_control import (
    ForceChirpControlConfig, LinearChirpProfile, FeedforwardResult,
    compute_feedforward_profile, fft_bin_amplitude, ConstantForceChirpController,
)

#%% Global variables

control_type = ControlTypes.CONSTANT_FORCE_CHIRP
maximum_name_length = 50


#%% Commands

class ConstantForceChirpCommands(Enum):
    START_CONTROL = 0
    STOP_CONTROL = 1


class ConstantForceChirpControlCommands(Enum):
    INITIALIZE_PARAMETERS = 0
    RUN_CONTROL = 1
    STOP_CONTROL = 2
    SHUTDOWN_ACHIEVED = 3


#%% Queues

class ConstantForceChirpQueues:
    """Container for the queues this environment manages, following the same
    shape as TransientQueues / RandomVibrationQueues."""

    def __init__(self,
                 environment_name: str,
                 environment_command_queue: VerboseMessageQueue,
                 gui_update_queue: Queue,
                 controller_communication_queue: VerboseMessageQueue,
                 data_in_queue: Queue,
                 data_out_queue: Queue,
                 log_file_queue: VerboseMessageQueue):
        self.environment_command_queue = environment_command_queue
        self.gui_update_queue = gui_update_queue
        self.data_analysis_command_queue = VerboseMessageQueue(log_file_queue, environment_name + ' Data Analysis Command Queue')
        self.signal_generation_command_queue = VerboseMessageQueue(log_file_queue, environment_name + ' Signal Generation Command Queue')
        self.spectral_command_queue = VerboseMessageQueue(log_file_queue, environment_name + ' Spectral Computation Command Queue')
        self.collector_command_queue = VerboseMessageQueue(log_file_queue, environment_name + ' Data Collector Command Queue')
        self.force_control_command_queue = VerboseMessageQueue(log_file_queue, environment_name + ' Force Control Command Queue')
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        # Sys-ID phase: collector -> spectral processing -> sys-id data analysis
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        # Control phase: collector -> force control process -> signal generation
        self.data_for_force_control_queue = mp.Queue()
        self.force_control_to_signal_generation_queue = mp.Queue()
        self.log_file_queue = log_file_queue


#%% Metadata

class ConstantForceChirpMetadata(AbstractSysIdMetadata):
    def __init__(self,
                 number_of_channels,
                 sample_rate,
                 output_channel_index,
                 force_channel_index,
                 acceleration_channel_indices,
                 target_force,
                 start_frequency,
                 end_frequency,
                 chirp_duration,
                 ramp_time,
                 max_voltage,
                 force_abort,
                 acceleration_abort,
                 control_update_rate,
                 amplitude_smoothing,
                 minimum_frequency,
                 maximum_frequency,
                 coherence_threshold,
                 regularization,
                 max_gain_step_per_update):
        super().__init__()
        self.number_of_channels = number_of_channels
        self.sample_rate = sample_rate
        self.output_channel_index = output_channel_index
        self.force_channel_index = force_channel_index
        self.acceleration_channel_indices = list(acceleration_channel_indices)
        self.target_force = target_force
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.chirp_duration = chirp_duration
        self.ramp_time = ramp_time
        self.max_voltage = max_voltage
        self.force_abort = force_abort
        self.acceleration_abort = acceleration_abort
        self.control_update_rate = control_update_rate
        self.amplitude_smoothing = amplitude_smoothing
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.coherence_threshold = coherence_threshold
        self.regularization = regularization
        self.max_gain_step_per_update = max_gain_step_per_update

    # --- AbstractSysIdMetadata contract ---
    # "Response" (sys-id) = force sensor + all monitored accelerometers, so a
    # single low-level identification sweep also yields informational H_aV(f)
    # for each accelerometer (useful for later modal analysis and for sanity
    # checking acceleration_abort), even though only the force row is used
    # for control.
    @property
    def number_of_channels(self):
        return self._number_of_channels

    @number_of_channels.setter
    def number_of_channels(self, value):
        self._number_of_channels = value

    @property
    def response_channel_indices(self):
        return [self.force_channel_index] + list(self.acceleration_channel_indices)

    @property
    def reference_channel_indices(self):
        return [self.output_channel_index]

    @property
    def response_transformation_matrix(self):
        return None

    @property
    def reference_transformation_matrix(self):
        return None

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def control_frame_size(self):
        """Samples per control-phase acquisition frame, from control_update_rate."""
        return max(4, int(round(self.sample_rate / self.control_update_rate)))

    def to_control_config(self) -> ForceChirpControlConfig:
        return ForceChirpControlConfig(
            target_force=self.target_force,
            start_frequency=self.start_frequency,
            end_frequency=self.end_frequency,
            chirp_duration=self.chirp_duration,
            ramp_time=self.ramp_time,
            max_voltage=self.max_voltage,
            force_abort=self.force_abort,
            acceleration_abort=self.acceleration_abort,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            coherence_threshold=self.coherence_threshold,
            regularization=self.regularization,
            control_update_rate=self.control_update_rate,
            amplitude_smoothing=self.amplitude_smoothing,
            max_gain_step_per_update=self.max_gain_step_per_update,
        )

    def store_to_netcdf(self, netcdf_group_handle: nc4._netCDF4.Group):
        super().store_to_netcdf(netcdf_group_handle)
        for field in ('output_channel_index', 'force_channel_index', 'target_force',
                      'start_frequency', 'end_frequency', 'chirp_duration', 'ramp_time',
                      'max_voltage', 'force_abort', 'acceleration_abort',
                      'control_update_rate', 'amplitude_smoothing', 'minimum_frequency',
                      'maximum_frequency', 'coherence_threshold', 'regularization',
                      'max_gain_step_per_update'):
            setattr(netcdf_group_handle, field, getattr(self, field))
        netcdf_group_handle.createDimension('acceleration_channels', len(self.acceleration_channel_indices))
        var = netcdf_group_handle.createVariable('acceleration_channel_indices', 'i4', ('acceleration_channels',))
        var[...] = self.acceleration_channel_indices


#%% UI
#
# Kept intentionally lean for a V1: parameter entry, start/stop, and a live
# force/voltage/frequency readout. It does not reproduce Transient's Excel
# template import, MDI plot-window tiling, or multi-tab prediction workflow
# -- those are natural follow-on additions once the control loop has been
# validated on the bench (see the hardware test plan), not prerequisites for
# a working, safe V1. The paired .ui files
# (components/constant_force_chirp_definition.ui,
# components/constant_force_chirp_run.ui) define exactly the widgets
# referenced below by name.

class ConstantForceChirpUI(AbstractSysIdUI):
    def __init__(self,
                 environment_name: str,
                 definition_tabwidget: QtWidgets.QTabWidget,
                 system_id_tabwidget: QtWidgets.QTabWidget,
                 test_predictions_tabwidget: QtWidgets.QTabWidget,
                 run_tabwidget: QtWidgets.QTabWidget,
                 environment_command_queue: VerboseMessageQueue,
                 controller_communication_queue: VerboseMessageQueue,
                 log_file_queue: Queue):
        super().__init__(environment_name,
                          environment_command_queue, controller_communication_queue, log_file_queue,
                          system_id_tabwidget)
        self.definition_widget = QtWidgets.QWidget()
        uic.loadUi(environment_definition_ui_paths[control_type], self.definition_widget)
        definition_tabwidget.addTab(self.definition_widget, self.environment_name)

        self.run_widget = QtWidgets.QWidget()
        uic.loadUi(environment_run_ui_paths[control_type], self.run_widget)
        run_tabwidget.addTab(self.run_widget, self.environment_name)

        self.physical_channel_names = None
        self.physical_output_indices = None

        self.definition_widget.output_channel_selector.currentIndexChanged.connect(self.update_channel_display)
        self.definition_widget.force_channel_selector.currentIndexChanged.connect(self.update_channel_display)
        self.run_widget.start_test_button.clicked.connect(self.start_control)
        self.run_widget.stop_test_button.clicked.connect(self.stop_control)

    def update_channel_display(self, *args):
        pass

    def initialize_data_acquisition(self, data_acquisition_parameters):
        super().initialize_data_acquisition(data_acquisition_parameters)
        self.physical_channel_names = ['{:} {:} {:}'.format(
            '' if channel.channel_type is None else channel.channel_type,
            channel.node_number, channel.node_direction)[:maximum_name_length]
            for channel in data_acquisition_parameters.channel_list]
        self.physical_output_indices = [i for i, channel in enumerate(data_acquisition_parameters.channel_list)
                                         if channel.feedback_device]
        for combo in (self.definition_widget.output_channel_selector,
                      self.definition_widget.force_channel_selector):
            combo.clear()
            combo.addItems(self.physical_channel_names)
        self.definition_widget.acceleration_channel_selector.clear()
        for name in self.physical_channel_names:
            item = QtWidgets.QListWidgetItem()
            item.setText(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.definition_widget.acceleration_channel_selector.addItem(item)

    @property
    def acceleration_channel_indices(self):
        widget = self.definition_widget.acceleration_channel_selector
        return [i for i in range(widget.count()) if widget.item(i).checkState() == Qt.Checked]

    @property
    def initialized_control_names(self):
        return ['Force'] + ['Accel {:}'.format(i + 1) for i in self.environment_parameters.acceleration_channel_indices]

    @property
    def initialized_output_names(self):
        return [self.physical_channel_names[self.environment_parameters.output_channel_index]]

    def collect_environment_definition_parameters(self) -> ConstantForceChirpMetadata:
        metadata = ConstantForceChirpMetadata(
            number_of_channels=len(self.physical_channel_names),
            sample_rate=self.data_acquisition_parameters.sample_rate,
            output_channel_index=self.definition_widget.output_channel_selector.currentIndex(),
            force_channel_index=self.definition_widget.force_channel_selector.currentIndex(),
            acceleration_channel_indices=self.acceleration_channel_indices,
            target_force=self.definition_widget.target_force_selector.value(),
            start_frequency=self.definition_widget.start_frequency_selector.value(),
            end_frequency=self.definition_widget.end_frequency_selector.value(),
            chirp_duration=self.definition_widget.chirp_duration_selector.value(),
            ramp_time=self.definition_widget.ramp_time_selector.value(),
            max_voltage=self.definition_widget.max_voltage_selector.value(),
            force_abort=self.definition_widget.force_abort_selector.value(),
            acceleration_abort=self.definition_widget.acceleration_abort_selector.value(),
            control_update_rate=self.definition_widget.control_update_rate_selector.value(),
            amplitude_smoothing=self.definition_widget.amplitude_smoothing_selector.value(),
            minimum_frequency=self.definition_widget.minimum_frequency_selector.value(),
            maximum_frequency=self.definition_widget.maximum_frequency_selector.value(),
            coherence_threshold=self.definition_widget.coherence_threshold_selector.value(),
            regularization=self.definition_widget.regularization_selector.value(),
            max_gain_step_per_update=self.definition_widget.max_gain_step_selector.value(),
        )
        self.update_sysid_metadata(metadata)
        return metadata

    def initialize_environment(self) -> ConstantForceChirpMetadata:
        self.environment_parameters = self.collect_environment_definition_parameters()
        return self.environment_parameters

    def retrieve_metadata(self, netcdf_handle):
        pass  # left for a follow-on pass; store_to_netcdf is implemented and sufficient for post-test analysis

    def create_environment_template(self, environment_name, workbook):
        pass  # Excel template import/export intentionally deferred for V1

    def set_parameters_from_template(self, worksheet):
        pass

    def start_control(self):
        self.environment_command_queue.put(self.environment_name, (ConstantForceChirpCommands.START_CONTROL, None))

    def stop_control(self):
        self.environment_command_queue.put(self.environment_name, (ConstantForceChirpCommands.STOP_CONTROL, None))

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return
        message, data = queue_data
        if message == 'force_control_update':
            t, frequency, target_force, measured_force, voltage, trim_gain, warnings = data
            self.run_widget.frequency_display.setValue(frequency)
            self.run_widget.measured_force_display.setValue(measured_force)
            self.run_widget.voltage_display.setValue(voltage)
            self.run_widget.trim_gain_display.setValue(trim_gain)
        elif message == 'force_control_aborted':
            reason = data
            QtWidgets.QMessageBox.warning(self.run_widget, 'Constant Force Chirp Aborted', reason)


#%% Force control process (Phase 2/3: feedforward + closed-loop trim)

class ConstantForceChirpControlProcess(AbstractMessageProcess):
    """Continuously drains newly acquired force/acceleration frames (routed
    here directly from the data collector's fan-out, bypassing spectral/CPSD
    processing entirely -- this environment does not need CPSD estimation
    during the control phase, only a per-frame force-amplitude estimate) and
    pushes updated drive-voltage amplitudes to signal generation.

    Structurally this plays the same role as
    ``RandomVibrationDataAnalysisProcess.run_control``
    (components/random_vibration_sys_id_data_analysis.py): a self-repeating
    command that pulls the newest available data and reacts to it. The
    difference is what it computes each update (a bounded scalar amplitude
    trim via ``control_laws.constant_force_chirp_control.ConstantForceChirpController``,
    instead of a CPSD-matrix pseudoinverse).
    """

    def __init__(self, process_name: str,
                 command_queue: VerboseMessageQueue,
                 data_in_queue: mp.queues.Queue,
                 data_out_queue: mp.queues.Queue,
                 environment_command_queue: VerboseMessageQueue,
                 gui_update_queue: mp.queues.Queue,
                 log_file_queue: mp.queues.Queue,
                 environment_name: str):
        super().__init__(process_name, log_file_queue, command_queue, gui_update_queue)
        self.map_command(ConstantForceChirpControlCommands.INITIALIZE_PARAMETERS, self.initialize_parameters)
        self.map_command(ConstantForceChirpControlCommands.RUN_CONTROL, self.run_control)
        self.map_command(ConstantForceChirpControlCommands.STOP_CONTROL, self.stop_control)
        self.environment_name = environment_name
        self.environment_command_queue = environment_command_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.controller = None
        self.sample_rate = None
        self.force_row = 0
        self.acceleration_rows = []
        self.running = False

    def initialize_parameters(self, data):
        config, feedforward, sample_rate, num_acceleration_channels = data
        profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
        self.controller = ConstantForceChirpController(config, feedforward, profile)
        self.sample_rate = sample_rate
        self.force_row = 0
        self.acceleration_rows = list(range(1, 1 + num_acceleration_channels))
        self.running = True
        flush_queue(self.data_in_queue)
        self.command_queue.put(self.process_name, (ConstantForceChirpControlCommands.RUN_CONTROL, None))

    def run_control(self, data):
        if not self.running:
            return
        frames = flush_queue(self.data_in_queue)
        for response_fft, reference_fft in frames:
            frequency_now = self.controller.chirp_profile.frequency(np.array([self._elapsed_time()]))[0]
            measured_force = fft_bin_amplitude(response_fft[self.force_row], self.sample_rate, frequency_now)
            acceleration_peaks = []
            for row in self.acceleration_rows:
                accel_time = np.fft.irfft(response_fft[row])
                acceleration_peaks.append(float(np.max(np.abs(accel_time))) if accel_time.size else 0.0)

            result = self.controller.step(t=self._elapsed_time(), measured_force=measured_force,
                                           acceleration_peaks=acceleration_peaks)
            self._advance_time()

            self.data_out_queue.put((result.voltage_amplitude,))
            self.gui_update_queue.put((self.environment_name, ('force_control_update',
                (self._elapsed_time(), frequency_now, self.controller.config.target_force,
                 result.measured_force, result.voltage_amplitude, result.trim_gain, result.warnings))))

            if result.aborted:
                self.running = False
                self.data_out_queue.put((0.0,))
                self.gui_update_queue.put((self.environment_name, ('force_control_aborted', result.abort_reason)))
                self.environment_command_queue.put(self.process_name,
                    (ConstantForceChirpCommands.STOP_CONTROL, result.abort_reason))
                break

        if self.running:
            self.command_queue.put(self.process_name, (ConstantForceChirpControlCommands.RUN_CONTROL, None))

    def _elapsed_time(self):
        return getattr(self, '_time', 0.0)

    def _advance_time(self):
        frame_duration = 1.0 / self.controller.config.control_update_rate
        self._time = self._elapsed_time() + frame_duration

    def stop_control(self, data):
        self.running = False
        flush_queue(self.data_in_queue)
        self.command_queue.flush(self.process_name)
        self.environment_command_queue.put(self.process_name,
            (ConstantForceChirpControlCommands.SHUTDOWN_ACHIEVED, None))


def constant_force_chirp_control_process(environment_name: str,
                                          command_queue: VerboseMessageQueue,
                                          data_in_queue: mp.queues.Queue,
                                          data_out_queue: mp.queues.Queue,
                                          environment_command_queue: VerboseMessageQueue,
                                          gui_update_queue: mp.queues.Queue,
                                          log_file_queue: mp.queues.Queue):
    instance = ConstantForceChirpControlProcess(
        environment_name + ' Force Control', command_queue, data_in_queue, data_out_queue,
        environment_command_queue, gui_update_queue, log_file_queue, environment_name)
    instance.run()


#%% Environment

class ConstantForceChirpEnvironment(AbstractSysIdEnvironment):

    def __init__(self,
                 environment_name: str,
                 queue_container: ConstantForceChirpQueues,
                 acquisition_active: mp.Value,
                 output_active: mp.Value):
        super().__init__(
            environment_name,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.controller_communication_queue,
            queue_container.log_file_queue,
            queue_container.collector_command_queue,
            queue_container.signal_generation_command_queue,
            queue_container.spectral_command_queue,
            queue_container.data_analysis_command_queue,
            queue_container.data_in_queue,
            queue_container.data_out_queue,
            acquisition_active,
            output_active)
        self.map_command(ConstantForceChirpCommands.START_CONTROL, self.start_control)
        self.map_command(ConstantForceChirpCommands.STOP_CONTROL, self.stop_environment)
        self.queue_container = queue_container
        self.frf = None
        self.coherence = None
        self.frequencies = None
        self.feedforward = None

    def system_id_complete(self, data):
        self.log('Finished System Identification')
        self.controller_communication_queue.put(
            self.environment_name, (GlobalCommands.COMPLETED_SYSTEM_ID, self.environment_name))
        (frames, avg, self.frequencies, self.frf, self.coherence,
         response_cpsd, reference_cpsd, condition,
         response_noise, reference_noise) = data
        # Row 0 of the response set is the force channel (see
        # ConstantForceChirpMetadata.response_channel_indices); the
        # remaining rows are the accelerometers, kept only for monitoring/
        # future modal-analysis use, not for control.
        force_frf = self.frf[:, 0, 0]
        force_coherence = self.coherence[:, 0] if self.coherence is not None else None
        config = self.environment_parameters.to_control_config()
        self.feedforward = compute_feedforward_profile(self.frequencies, force_frf, config,
                                                         coherence=force_coherence)
        self.gui_update_queue.put((self.environment_name, ('sysid_complete', None)))

    def get_signal_generation_metadata(self):
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=1,  # amplitude ramp is handled inside the controller (config.ramp_time)
            output_transformation_matrix=None,
        )

    def get_control_data_collector_metadata(self) -> CollectorMetadata:
        return CollectorMetadata(
            num_channels=self.environment_parameters.number_of_channels,
            response_channel_indices=self.environment_parameters.response_channel_indices,
            reference_channel_indices=self.environment_parameters.reference_channel_indices,
            acquisition_type=AcquisitionType.FREE_RUN,
            acceptance=Acceptance.AUTOMATIC,
            acceptance_function=None,
            overlap_fraction=0,
            trigger_channel_index=0,
            trigger_slope=TriggerSlope.POSITIVE,
            trigger_level=0,
            trigger_hysteresis=0,
            trigger_hysteresis_samples=0,
            pretrigger_fraction=0,
            frame_size=self.environment_parameters.control_frame_size,
            # Required so fft_bin_amplitude's inverse-FFT reconstruction is
            # exact -- see the module docstring and
            # control_laws.constant_force_chirp_control.fft_bin_amplitude.
            window=Window.RECTANGLE,
        )

    def start_control(self, data):
        if self.feedforward is None:
            self.gui_update_queue.put(('error', ('Perform System Identification',
                'Run system identification before starting the constant-force chirp')))
            return
        self.log('Starting Constant Force Chirp')
        config = self.environment_parameters.to_control_config()

        # Force control process: computes the closed-loop voltage trim
        self.queue_container.force_control_command_queue.put(
            self.environment_name,
            (ConstantForceChirpControlCommands.INITIALIZE_PARAMETERS,
             (config, self.feedforward, self.data_acquisition_parameters.sample_rate,
              len(self.environment_parameters.acceleration_channel_indices))))

        # Signal generation: start with the feedforward voltage at the sweep's start frequency
        initial_level = float(np.interp(config.start_frequency, self.feedforward.frequencies,
                                         self.feedforward.voltage_amplitude))
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_PARAMETERS, self.get_signal_generation_metadata()))
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
             ConstantForceChirpSignalGenerator(
                 level=initial_level,
                 sample_rate=self.data_acquisition_parameters.sample_rate,
                 num_samples_per_frame=self.data_acquisition_parameters.samples_per_write,
                 num_signals=1,
                 start_frequency=config.start_frequency,
                 end_frequency=config.end_frequency,
                 chirp_duration=config.chirp_duration,
                 output_oversample=self.data_acquisition_parameters.output_oversample)))
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.SET_TEST_LEVEL, 1.0))
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None))

        # Data collector: acquire force + acceleration frames for the control loop
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR, self.get_control_data_collector_metadata()))
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.SET_TEST_LEVEL, (0, 1)))
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None))

    def stop_environment(self, data):
        self.log('Stopping Constant Force Chirp' + ('' if data is None else ': {:}'.format(data)))
        self.queue_container.force_control_command_queue.put(
            self.environment_name, (ConstantForceChirpControlCommands.STOP_CONTROL, None))
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None))
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.STOP, None))


#%% Process

def constant_force_chirp_process(environment_name: str,
                                  input_queue: VerboseMessageQueue,
                                  gui_update_queue: Queue,
                                  controller_communication_queue: VerboseMessageQueue,
                                  log_file_queue: Queue,
                                  data_in_queue: Queue,
                                  data_out_queue: Queue,
                                  acquisition_active: mp.Value,
                                  output_active: mp.Value):
    queue_container = ConstantForceChirpQueues(
        environment_name, input_queue, gui_update_queue, controller_communication_queue,
        data_in_queue, data_out_queue, log_file_queue)

    spectral_proc = mp.Process(target=spectral_processing_process,
                                args=(environment_name,
                                      queue_container.spectral_command_queue,
                                      queue_container.data_for_spectral_computation_queue,
                                      queue_container.updated_spectral_quantities_queue,
                                      queue_container.environment_command_queue,
                                      queue_container.gui_update_queue,
                                      queue_container.log_file_queue))
    spectral_proc.start()
    analysis_proc = mp.Process(target=sysid_data_analysis_process,
                                args=(environment_name,
                                      queue_container.data_analysis_command_queue,
                                      queue_container.updated_spectral_quantities_queue,
                                      queue_container.force_control_to_signal_generation_queue,  # unused during sys-id
                                      queue_container.environment_command_queue,
                                      queue_container.gui_update_queue,
                                      queue_container.log_file_queue))
    analysis_proc.start()
    force_control_proc = mp.Process(target=constant_force_chirp_control_process,
                                     args=(environment_name,
                                           queue_container.force_control_command_queue,
                                           queue_container.data_for_force_control_queue,
                                           queue_container.force_control_to_signal_generation_queue,
                                           queue_container.environment_command_queue,
                                           queue_container.gui_update_queue,
                                           queue_container.log_file_queue))
    force_control_proc.start()
    siggen_proc = mp.Process(target=signal_generation_process,
                              args=(environment_name,
                                    queue_container.signal_generation_command_queue,
                                    queue_container.force_control_to_signal_generation_queue,
                                    queue_container.data_out_queue,
                                    queue_container.environment_command_queue,
                                    queue_container.log_file_queue,
                                    queue_container.gui_update_queue))
    siggen_proc.start()
    collection_proc = mp.Process(target=data_collector_process,
                                  args=(environment_name,
                                        queue_container.collector_command_queue,
                                        queue_container.data_in_queue,
                                        # Fixed fan-out for the process's lifetime: sys-id phase frames
                                        # go to spectral processing, control-phase frames go to the
                                        # force control process. Each phase's collector metadata
                                        # (re-initialized via FORCE_INITIALIZE_COLLECTOR between phases)
                                        # determines frame size/window/channels; the consumer that isn't
                                        # currently running simply leaves its queue unread until flushed.
                                        [queue_container.data_for_spectral_computation_queue,
                                         queue_container.data_for_force_control_queue],
                                        queue_container.environment_command_queue,
                                        queue_container.log_file_queue,
                                        queue_container.gui_update_queue))
    collection_proc.start()

    process_class = ConstantForceChirpEnvironment(
        environment_name, queue_container, acquisition_active, output_active)
    process_class.run()

    process_class.log('Joining Subprocesses')
    spectral_proc.join()
    analysis_proc.join()
    force_control_proc.join()
    siggen_proc.join()
    collection_proc.join()

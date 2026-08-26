# -*- coding: utf-8 -*-
"""
Plots the diagnostics logged by the Sine Force Control environment
(see components/sine_force_control_environment.py) from a saved NetCDF file.

Usage
-----
    python plot_sine_force_control_diagnostics.py path/to/diagnostics.nc4
    python plot_sine_force_control_diagnostics.py path/to/diagnostics.nc4 --group "Sine Force Control"

If --group is omitted, the first (and normally only) group in the file is used.
"""
import argparse
import sys
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt


def load_diagnostics(filename: str, group_name: str = None):
    dataset = nc4.Dataset(filename, 'r')
    if group_name is None:
        group_name = list(dataset.groups.keys())[0]
    group = dataset.groups[group_name]

    data = {
        'time': group.variables['time'][:].filled(np.nan),
        'frequency': group.variables['instantaneous_frequency'][:].filled(np.nan),
        'target': group.variables['force_target'][:].filled(np.nan),
        'measured': group.variables['force_amplitude_measured'][:].filled(np.nan),
        'relative_error': group.variables['relative_force_error'][:].filled(np.nan),
        'drive': group.variables['drive_amplitude_command'][:].filled(np.nan),
        'state': np.array(group.variables['controller_state'][:]),
        'saturated': group.variables['controller_saturated'][:].filled(0).astype(bool),
        'valid': group.variables['estimator_valid'][:].filled(0).astype(bool),
    }
    metadata = {
        'target_force': group.target_force,
        'max_drive_v': group.max_drive_v,
        'abort_drive_v': group.abort_drive_v,
        'sweep_type': group.sweep_type,
        'f_start': group.f_start,
        'f_stop': group.f_stop,
    }
    dataset.close()
    return data, metadata, group_name


def plot_diagnostics(data, metadata, group_name):
    valid = data['valid']
    saturated = data['saturated']

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    fig.suptitle('Sine Force Control Diagnostics -- {:}'.format(group_name))

    ax = axes[0]
    ax.plot(data['time'], data['frequency'], color='tab:blue')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(data['time'], data['target'], '--', color='tab:gray', label='Target')
    ax.plot(data['time'][valid], data['measured'][valid], '.', color='tab:blue',
            markersize=3, label='Measured (valid)')
    ax.plot(data['time'][~valid], data['measured'][~valid], '.', color='tab:red',
            markersize=3, label='Measured (invalid)')
    ax.set_ylabel('Force (N peak)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(data['time'][valid], data['relative_error'][valid] * 100.0, '.',
            color='tab:blue', markersize=3)
    ax.axhline(0.0, color='k', linewidth=0.5)
    ax.set_ylabel('Relative Force\nError (%)')
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(data['time'], data['drive'], color='tab:green', label='Drive Amplitude')
    ax.plot(data['time'][saturated], data['drive'][saturated], 'x', color='tab:orange',
            markersize=5, label='Saturated')
    ax.axhline(metadata['max_drive_v'], color='tab:orange', linestyle='--', linewidth=1,
               label='Max Drive (control limit)')
    ax.axhline(metadata['abort_drive_v'], color='tab:red', linestyle='--', linewidth=1,
               label='Abort Drive')
    ax.set_ylabel('Drive (V peak)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Second figure: force vs frequency, the most diagnostic view for a sweep.
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(data['frequency'][valid], data['measured'][valid], '.', color='tab:blue',
             markersize=3, label='Measured Force')
    ax2.axhline(metadata['target_force'], color='tab:gray', linestyle='--',
                label='Target Force')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Force (N peak)')
    ax2.set_title('Measured Force vs. Frequency -- {:}'.format(group_name))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()


if __name__ == '__main__':
    print('DEBUG sys.argv:', sys.argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', help='Path to the .nc4 diagnostics file')
    parser.add_argument('--group', default=None,
                        help='Environment group name (default: first group in the file)')
    args = parser.parse_args()

    data, metadata, group_name = load_diagnostics(args.filename, args.group)
    print('Loaded {:} samples from group "{:}"'.format(len(data['time']), group_name))
    print('Sweep: {:} from {:.2f} to {:.2f} Hz, target force {:.3f} N peak'.format(
        metadata['sweep_type'], metadata['f_start'], metadata['f_stop'],
        metadata['target_force']))
    plot_diagnostics(data, metadata, group_name)

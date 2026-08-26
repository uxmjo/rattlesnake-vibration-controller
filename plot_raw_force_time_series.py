# -*- coding: utf-8 -*-
"""
Plots the RAW, sample-by-sample force channel time signal from a Rattlesnake
global streaming file (created via the main GUI's Streaming section -- NOT
the Sine Force Control environment's own diagnostics file, which only
contains the per-control-update amplitude estimate).

This reads the schema written by components/streaming.py:
    - attribute 'sample_rate'
    - variable 'time_data', shape (response_channels, time_samples)
    - per-channel metadata under /channels/<field> (e.g. channel_type,
      node_number, node_direction, feedback_device), indexed the same way
      as the rows of 'time_data'

Usage
-----
    # Auto-detect the force channel (looks for channel_type containing "force"
    # among channels with no feedback_device, i.e. measurement channels):
    python plot_raw_force_time_series.py path/to/streaming_file.nc4

    # List all channels in the file (to find the right index if
    # auto-detection doesn't find your force channel):
    python plot_raw_force_time_series.py path/to/streaming_file.nc4 --list-channels

    # Explicitly pick a channel by its row index in the channel table:
    python plot_raw_force_time_series.py path/to/streaming_file.nc4 --channel 0

    # Only plot a time window (in seconds) instead of the whole recording:
    python plot_raw_force_time_series.py path/to/streaming_file.nc4 --start 5 --stop 10
"""
import argparse
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt


def list_channels(dataset: nc4.Dataset):
    node_number = dataset['/channels/node_number'][:]
    node_direction = dataset['/channels/node_direction'][:]
    channel_type = dataset['/channels/channel_type'][:]
    unit = dataset['/channels/unit'][:]
    feedback_device = dataset['/channels/feedback_device'][:]
    print('{:>4}  {:>10}  {:>10}  {:>10}  {:>8}  {:}'.format(
        'idx', 'node', 'direction', 'type', 'unit', 'role'))
    for i in range(len(node_number)):
        role = 'output' if feedback_device[i] else 'measurement'
        print('{:>4}  {:>10}  {:>10}  {:>10}  {:>8}  {:}'.format(
            i, node_number[i], node_direction[i], channel_type[i], unit[i], role))


def find_force_channel(dataset: nc4.Dataset) -> int:
    channel_type = dataset['/channels/channel_type'][:]
    feedback_device = dataset['/channels/feedback_device'][:]
    candidates = [i for i, (ct, fb) in enumerate(zip(channel_type, feedback_device))
                  if not fb and 'force' in str(ct).lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise ValueError(
            'Could not auto-detect a Force channel (no measurement channel with '
            '"Force" in its Channel Type). Run with --list-channels and pass '
            '--channel <index> explicitly.')
    raise ValueError(
        'Multiple candidate Force channels found: {:}. Pass --channel <index> '
        'explicitly.'.format(candidates))


def load_force_time_series(filename: str, channel: int = None,
                           start: float = None, stop: float = None):
    dataset = nc4.Dataset(filename, 'r')
    sample_rate = dataset.sample_rate
    if channel is None:
        channel = find_force_channel(dataset)
    node_number = dataset['/channels/node_number'][channel]
    node_direction = dataset['/channels/node_direction'][channel]
    unit = dataset['/channels/unit'][channel]

    n_samples = dataset.dimensions['time_samples'].size
    start_sample = 0 if start is None else max(0, int(start * sample_rate))
    stop_sample = n_samples if stop is None else min(n_samples, int(stop * sample_rate))

    force = dataset.variables['time_data'][channel, start_sample:stop_sample]
    force = np.asarray(force, dtype=float)
    time = (start_sample + np.arange(force.size)) / sample_rate

    dataset.close()
    return time, force, channel, node_number, node_direction, unit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename', help='Path to the streaming .nc4 file')
    parser.add_argument('--channel', type=int, default=None,
                        help='Channel table row index to plot (default: auto-detect Force channel)')
    parser.add_argument('--list-channels', action='store_true',
                        help='List all channels in the file and exit')
    parser.add_argument('--start', type=float, default=None, help='Start time in seconds')
    parser.add_argument('--stop', type=float, default=None, help='Stop time in seconds')
    args = parser.parse_args()

    if args.list_channels:
        with nc4.Dataset(args.filename, 'r') as ds:
            list_channels(ds)
    else:
        time, force, channel, node_number, node_direction, unit = load_force_time_series(
            args.filename, args.channel, args.start, args.stop)
        print('Plotting channel {:} (Node {:}{:}), {:} samples, unit={:}'.format(
            channel, node_number, node_direction, force.size, unit))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time, force, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force ({:})'.format(unit))
        ax.set_title('Raw Force Time Series -- Channel {:} (Node {:}{:})'.format(
            channel, node_number, node_direction))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

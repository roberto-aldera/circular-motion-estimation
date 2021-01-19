import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
from pose_tools.pose_utils import *
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


def plot_odometries(params, radar_state_mono):
    figure_path = params.input_path + "figs_odometry/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    se3s, timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    # print(se3s[0])

    k_start_index_from_odometry = 50
    ro_se3s = []
    ro_timestamps = []

    for i in range(params.num_samples):
        pb_state, name_scan, _ = radar_state_mono[i]
        ro_state = get_ro_state_from_pb(pb_state)

        relative_pose_index = i + k_start_index_from_odometry + 1
        relative_pose_timestamp = timestamps[relative_pose_index]

        # ensure timestamps are within a reasonable limit of each other (microseconds)
        assert (ro_state.timestamp - relative_pose_timestamp) < 500

        # Get motion estimate that was calculated from RO
        ro_se3, ro_timestamp = get_poses_from_serialised_transform(ro_state.g_motion_estimate)
        ro_se3s.append(ro_se3)
        ro_timestamps.append(ro_timestamp)

    start_time_offset = ro_timestamps[0]
    ro_time_seconds = [(x - start_time_offset) / 1e6 for x in ro_timestamps[1:]]
    ro_x, ro_y, ro_th = get_x_y_th_velocities_from_poses(ro_se3s, ro_timestamps)

    gt_start_time_offset = timestamps[0]
    gt_time_seconds = [(x - gt_start_time_offset) / 1e6 for x in timestamps[1:]]
    gt_x, gt_y, gt_th = get_x_y_th_velocities_from_poses(se3s, timestamps)
    gt_x = gt_x[k_start_index_from_odometry:]

    plt.figure(figsize=(15, 5))
    dim = params.num_samples
    # plt.xlim(0, dim)
    plt.grid()
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(ro_time_seconds, ro_x, '.-', label="ro_x")
    plt.plot(ro_time_seconds, ro_y, '.-', label="ro_y")
    plt.plot(ro_time_seconds, ro_th, '.-', label="ro_th")
    plt.plot(gt_time_seconds[:dim], gt_x[:dim], '.-', label="gt_x")
    plt.plot(gt_time_seconds[:dim], gt_y[:dim], '.-', label="gt_y")
    plt.plot(gt_time_seconds[:dim], gt_th[:dim], '.-', label="gt_th")
    plt.title("Pose estimates: RO vs ground-truth")
    plt.xlabel("Time (s)")
    plt.ylabel("units/s")
    plt.legend()
    plt.savefig("%s%s" % (output_path, "/odometry_comparison.pdf"))
    plt.close()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running plotting...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    # get a landmark set in and plot it
    plot_odometries(params, radar_state_mono)


if __name__ == "__main__":
    main()

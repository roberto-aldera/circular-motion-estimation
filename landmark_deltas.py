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


def find_landmark_deltas(params, radar_state_mono):
    figure_path = params.input_path + "figs_delta_landmarks/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    se3s, timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    # print(se3s[0])
    global_pose = np.eye(4)

    plt.figure(figsize=(20, 20))
    # dim = 200
    # plt.xlim(-dim, dim)
    # plt.ylim(-dim, dim)

    k_every_nth_scan = 1
    k_start_index_from_odometry = 140

    for i in range(params.num_samples):
        pb_state, name_scan, _ = radar_state_mono[i]
        ro_state = get_ro_state_from_pb(pb_state)
        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        primary_landmarks = np.c_[
            primary_landmarks, np.ones(len(primary_landmarks))]  # so that se3 multiplication works

        relative_pose_index = i + k_start_index_from_odometry + 1
        relative_pose_timestamp = timestamps[relative_pose_index]

        # ensure timestamps are within a reasonable limit of each other (microseconds)
        assert (ro_state.timestamp - relative_pose_timestamp) < 500

        # global_pose = global_pose @ se3s[relative_pose_index]
        primary_landmarks = np.transpose(global_pose @ np.transpose(primary_landmarks))

        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))
        eigenvector = get_matrix_from_pb(ro_state.eigen_vector)

        # Get the best matches in order from the unary candidates using the eigenvector elements
        unary_matches = get_matrix_from_pb(ro_state.unary_match_candidates).astype(int)
        unary_matches = np.reshape(unary_matches, (unary_matches.shape[1], -1))

        print("Size of primary landmarks:", len(primary_landmarks))
        print("Size of secondary landmarks:", len(secondary_landmarks))
        k_match_ratio = 0.4

        best_matches = np.empty((int(unary_matches.shape[0] * k_match_ratio), 2))
        num_matches = 0
        while num_matches < len(best_matches):
            eigen_max_idx = np.argmax(eigenvector)
            proposed_new_match = unary_matches[eigen_max_idx].astype(int)
            # do a check here, to see if this proposed match from the unary candidates is a duplicate
            if not (best_matches[:, 1] == proposed_new_match[1]).any():
                # then this is not a duplicate
                best_matches[num_matches] = proposed_new_match
                num_matches += 1
            eigenvector[eigen_max_idx] = 0  # set to zero, so that next time we seek the max it'll be the next match

        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        matches_to_plot = best_matches.astype(int)

        if i % k_every_nth_scan == 0:
            print("Processing index: ", i)
            # plot x and y swapped around so that robot is moving forward as upward direction
            p1 = plt.plot(primary_landmarks[:, 1], primary_landmarks[:, 0], '+', markerfacecolor='none', markersize=1)
            p2 = plt.plot(secondary_landmarks[:, 1], secondary_landmarks[:, 0], '+', markerfacecolor='none',
                          markersize=1)
            for match_idx in range(len(matches_to_plot)):
                x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
                y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
                x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
                y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
                # print("landmark 1 point 'x'-coord:", x1)
                # print("landmark 2 point 'x'-coord:", x2)
                # print("landmark 1 point 'y'-coord:", y1)
                # print("landmark 2 point 'y'-coord:", y2)
                plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1 - (match_idx / len(matches_to_plot)))

    # plot sensor range for Oxford radar robotcar dataset
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    r = 163
    x1 = global_pose[1, 3] + r * np.cos(circle_theta)
    x2 = global_pose[0, 3] + r * np.sin(circle_theta)
    plt.plot(x1, x2, 'g--')

    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("%s%s%s" % (output_path, "/delta_landmarks", ".png"))
    plt.savefig("%s%s%s" % (output_path, "/delta_landmarks", ".pdf"))
    plt.close()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--relative_poses', type=str, default="", help='Path to relative pose file')
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running delta landmarks...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    # get a landmark set in and plot it
    find_landmark_deltas(params, radar_state_mono)


if __name__ == "__main__":
    main()

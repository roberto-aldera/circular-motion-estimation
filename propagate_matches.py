import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
import pandas as pd
import pdb
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


@dataclass
class Landmark:
    scan_idx: int
    x_pos: int
    y_pos: int


def propagate_matches(params, radar_state_mono):
    figure_path = params.input_path + "figs_match_propagation/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    se3s, timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    # print(se3s[0])
    global_pose = np.eye(4)

    k_every_nth_scan = 1
    k_start_index_from_odometry = 140

    df_match_chains = pd.DataFrame()

    for i in range(params.num_samples):
        plt.figure(figsize=(20, 20))
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
        k_match_ratio = 0.5  # this is an upper limit, it's a bit more crude than what we do in RO
        max_matches = int(unary_matches.shape[0] * k_match_ratio)  # len(unary_matches) * k_match_ratio
        size_of_smaller_landmark_set = min(len(primary_landmarks), len(secondary_landmarks))

        best_matches = np.zeros((max_matches, 2))
        match_weight = np.zeros(best_matches.shape[0])

        num_matches = 0
        search_idx = 0
        while search_idx < size_of_smaller_landmark_set:
            eigen_max_idx = np.argmax(eigenvector)
            max_eigenvector_val = eigenvector[eigen_max_idx]
            if max_eigenvector_val ** 2 < (
                    1.0 / size_of_smaller_landmark_set) or num_matches >= max_matches or max_eigenvector_val < 0:
                print("So this happened...", search_idx)
                print(max_eigenvector_val ** 2 < (1.0 / size_of_smaller_landmark_set))
                print(num_matches > max_matches)
                print(max_eigenvector_val < 0)
                break
            search_idx += 1
            proposed_new_match = unary_matches[eigen_max_idx].astype(int)
            # do a check here, to see if this proposed match from the unary candidates is a duplicate

            if not (proposed_new_match[0] == [point[0] for point in best_matches]).any():
                # then this is not a duplicate
                best_matches[num_matches] = proposed_new_match
                match_weight[num_matches] = eigenvector[eigen_max_idx]
                num_matches += 1
            else:
                print("Duplicate skipped: ", proposed_new_match)
            eigenvector[eigen_max_idx] = 0  # set to zero, so that next time we seek the max it'll be the next match

        # clean up to remove zeros (had to be initialised when we didn't yet know their final size)
        best_matches = best_matches[~np.all(best_matches == 0, axis=1)]
        match_weight = match_weight[~np.all(match_weight == 0, axis=0)][0]
        normalised_match_weight = match_weight / match_weight[0]
        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        # matches_to_plot = best_matches.astype(int)
        matches_to_plot = selected_matches.astype(int)

        if df_match_chains.empty:
            print("Filling match chain container with first match pairs...")
            df_match_chains = pd.DataFrame(pd.NA, index=range(0, len(matches_to_plot)), columns=range(0, 6))
            # fill first and second columns with initial matches
            df_match_chains.at[:, 0:1] = matches_to_plot
        else:
            print("Appending matches...")
            # iterate through points that were matched in the previous scan
            last_matched_points = np.array(df_match_chains[i])

            for match_idx in range(len(last_matched_points)):
                # then check if they are present in the new matches
                if pd.isna(last_matched_points[match_idx]) is False:
                    idx = np.where(matches_to_plot[:, 0] == last_matched_points[match_idx])
                    print(idx)
                    if matches_to_plot[idx, 1]:
                        match_to_add = matches_to_plot[idx, 1].item()
                        df_match_chains.at[match_idx, i + 1] = match_to_add

        print(df_match_chains)
        print("Persistence so far:", df_match_chains.count())
        # pdb.set_trace()

        if i % k_every_nth_scan == 0:
            print("Processing index: ", i)
            # plot x and y swapped around so that robot is moving forward as upward direction
            p1 = plt.plot(primary_landmarks[:, 1], primary_landmarks[:, 0], '+', markerfacecolor='none', markersize=1)
            p2 = plt.plot(secondary_landmarks[:, 1], secondary_landmarks[:, 0], '+', markerfacecolor='none',
                          markersize=1)
            landmarks = []
            for match_idx in range(len(matches_to_plot)):
                x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
                y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
                x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
                y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
                plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=normalised_match_weight[match_idx])
                landmarks.append(Landmark(i, x1, x2))
                # landmark = Landmark(i, x1, x2)

        # plot sensor range for Oxford radar robotcar dataset
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        r = 163
        x1 = global_pose[1, 3] + r * np.cos(circle_theta)
        x2 = global_pose[0, 3] + r * np.sin(circle_theta)
        plt.plot(x1, x2, 'g--')

        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("%s%s%i%s" % (output_path, "/matches", i, ".pdf"))
        plt.close()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--relative_poses', type=str, default="", help='Path to relative pose file')
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running match propagation...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    # get a landmark set in and plot it
    propagate_matches(params, radar_state_mono)


if __name__ == "__main__":
    main()

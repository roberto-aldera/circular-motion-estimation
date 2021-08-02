# Script for getting motion estimate from RO using all the matches

# python full_matches_only.py --input_path /workspace/data/ro-state-files/radar_oxford_10k/2019-01-10-11-46-21/ --output_path /workspace/data/landmark-distortion/final-results/2019-01-10-11-46-21/ --num_samples 10000


import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
import pdb
from tqdm import tqdm
from pyslam.metrics import TrajectoryMetrics
from pose_tools.pose_utils import *
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb
from get_rigid_body_motion import get_motion_estimate_from_svd

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


def ro_motion_estimation(params, radar_state_mono):
    # ro_se3s = []
    # ro_timestamps = []

    poses_from_full_match_set = []
    timestamps_from_ro_state = []

    num_iterations = min(params.num_samples, len(radar_state_mono))
    print("Running for", num_iterations, "samples")

    for i in tqdm(range(num_iterations)):
        pb_state, name_scan, _ = radar_state_mono[i]
        ro_state = get_ro_state_from_pb(pb_state)
        timestamps_from_ro_state.append(ro_state.timestamp)

        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        primary_landmarks = np.c_[
            primary_landmarks, np.ones(len(primary_landmarks))]  # so that se3 multiplication works

        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))

        # print("Size of primary landmarks:", len(primary_landmarks))
        # print("Size of secondary landmarks:", len(secondary_landmarks))

        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        matches_to_plot = selected_matches.astype(int)

        # print("Processing index: ", i)
        matched_points = []
        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            # plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
            matched_points.append([x1, x2, y1, y2])

        # Motion estimate from running SVD on all the points
        P1 = []
        P2 = []
        for match in matched_points:
            x1 = match[0]
            x2 = match[1]
            y1 = match[2]
            y2 = match[3]
            P1.append([x1, y1])
            P2.append([x2, y2])
        P1 = np.transpose(P1)
        P2 = np.transpose(P2)
        v, theta_R = get_motion_estimate_from_svd(P1, P2, weights=np.ones(P1.shape[1]))
        pose_from_svd = [v[1], v[0], -theta_R]  # this line applies the transform to get into the robot frame
        poses_from_full_match_set.append(pose_from_svd)

        # print("SVD motion estimate (x, y, th):", pose_from_svd)

    # x_vals_full_match_set = [pose[1] for pose in poses_from_full_match_set]
    # y_vals_full_match_set = [pose[0] for pose in poses_from_full_match_set]

    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_full_match_set,
                                      pose_source="full_matches",
                                      export_folder=params.output_path)


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--output_path', type=str, default="",
                        help='Path to folder where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running script...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    ro_motion_estimation(params, radar_state_mono)


if __name__ == "__main__":
    main()

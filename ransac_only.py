# Script for getting motion estimate by running ransac on the matches to reject outliers
# Runs SVD over inliers from the best RANSAC model compared to over all matches
# This is an update to the previous ransac_poses.py script

# python ransac_only.py --input_path /workspace/data/landmark-distortion/ro_state_pb_developing/ro_state_files/ --output_path /workspace/data/landmark-distortion/RANSAC-baseline/ --num_samples 80


import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
import pdb
import csv
import time
from tqdm import tqdm
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb
from get_rigid_body_motion import get_motion_estimate_from_svd
from ransac_utilities import get_pose_estimates_with_ransac, get_best_ransac_motion_estimate_index, \
    plot_points_and_match_errors, get_all_inliers_from_best_ransac_motion_estimate

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


def ransac_motion_estimation(params, radar_state_mono, results_path):
    poses_from_inliers = []
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

        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        matches_to_plot = selected_matches.astype(int)

        P1 = []
        P2 = []
        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            P1.append([x1, y1])
            P2.append([x2, y2])
        P1 = np.transpose(P1)
        P2 = np.transpose(P2)
        # Motion estimate from running SVD on all the points
        # v, theta_R = get_motion_estimate_from_svd(P1, P2, weights=np.ones(P1.shape[1]))
        # pose_from_svd = [v[1], v[0], -theta_R]  # this line applies the transform to get into the robot frame
        # poses_from_full_match_set.append(pose_from_svd)
        # print("SVD motion estimate (x, y, th):", pose_from_svd)

        # Running SVD on all inliers from the best RANSAC model
        # I think this was 100 iterations, it's slower but significantly improves accuracy
        pose_estimates = get_pose_estimates_with_ransac(P1, P2, iterations=50)
        champion_inliers = get_all_inliers_from_best_ransac_motion_estimate(P1, P2, pose_estimates,
                                                                            inlier_threshold=0.01)

        P1_inliers = P1[:, champion_inliers]
        P2_inliers = P2[:, champion_inliers]
        v, theta_R = get_motion_estimate_from_svd(P1_inliers, P2_inliers, weights=np.ones(P1_inliers.shape[1]))
        pose_from_svd = [v[1], v[0], -theta_R]  # this line applies the transform to get into the robot frame
        poses_from_inliers.append(pose_from_svd)
        # print("Pose from SVD on best inliers:", pose_from_svd)

    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_inliers,
                                      pose_source="ransac",
                                      export_folder=results_path)


def save_timestamps_and_x_y_th_to_csv(timestamps, x_y_th, pose_source, export_folder):
    # Save poses with format: timestamp, dx, dy, dth
    with open("%s%s%s" % (export_folder, pose_source, "_poses.csv"), 'w') as poses_file:
        wr = csv.writer(poses_file, delimiter=",")
        x_values = [item[0] for item in x_y_th]
        y_values = [item[1] for item in x_y_th]
        th_values = [item[2] for item in x_y_th]
        for idx in range(len(timestamps)):
            timestamp_and_x_y_th = [timestamps[idx], x_values[idx], y_values[idx], th_values[idx]]
            wr.writerow(timestamp_and_x_y_th)


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default=settings.RO_STATE_PATH,
                        help='Path to folder containing required inputs')
    parser.add_argument('--output_path', type=str, default=settings.POSE_OUTPUT_PATH,
                        help='Path to folder where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running script...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    results_path = settings.POSE_OUTPUT_PATH + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    ransac_motion_estimation(params, radar_state_mono, results_path)


if __name__ == "__main__":
    main()

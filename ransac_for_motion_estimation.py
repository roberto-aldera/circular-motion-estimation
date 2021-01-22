# Script for getting motion estimate by running ransac on the matches to reject outliers
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
from pose_tools.pose_utils import *
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb
from get_rigid_body_motion import get_motion_estimate_from_svd
from ransac_utilities import get_pose_estimates_with_ransac, get_best_ransac_motion_estimate_index

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


def ransac_motion_estimation(params, radar_state_mono):
    figure_path = params.input_path + "figs_ransac_motion_estimation/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    for i in range(params.num_samples):
        plt.figure(figsize=(20, 20))
        # dim = 200
        # plt.xlim(-dim, dim)
        # plt.ylim(-dim, dim)
        pb_state, name_scan, _ = radar_state_mono[i]
        ro_state = get_ro_state_from_pb(pb_state)
        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        primary_landmarks = np.c_[
            primary_landmarks, np.ones(len(primary_landmarks))]  # so that se3 multiplication works

        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))

        print("Size of primary landmarks:", len(primary_landmarks))
        print("Size of secondary landmarks:", len(secondary_landmarks))

        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        matches_to_plot = selected_matches.astype(int)

        print("Processing index: ", i)
        # plot x and y swapped around so that robot is moving forward as upward direction
        plt.plot(primary_landmarks[:, 1], primary_landmarks[:, 0], '+', markerfacecolor='none', markersize=1,
                 color="tab:blue")
        plt.plot(secondary_landmarks[:, 1], secondary_landmarks[:, 0], '+', markerfacecolor='none',
                 markersize=1, color="tab:orange")
        matched_points = []
        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
            matched_points.append([x1, x2, y1, y2])

        # plot sensor range for Oxford radar robotcar dataset
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        r = 163
        circle_x1 = 0 + r * np.cos(circle_theta)
        circle_x2 = 0 + r * np.sin(circle_theta)
        plt.plot(circle_x1, circle_x2, 'g--')

        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("%s%s%i%s" % (output_path, "/landmark_matches_", i, ".pdf"))
        plt.close()

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
        pose_from_svd = [v[0], v[1], theta_R]
        T_model = np.transpose(np.array([pose_from_svd[0:2]]))
        theta_model = pose_from_svd[2]

        R_model = np.array([[np.cos(theta_model), -np.sin(theta_model)], [np.sin(theta_model), np.cos(theta_model)]])
        P_model = R_model @ P1 + T_model

        matched_points = []
        plt.figure(figsize=(10, 10))
        plt.plot(P1[0], P1[1], '+', markerfacecolor='none', markersize=1, color="tab:blue", label="Live")
        plt.plot(P2[0], P2[1], '+', markerfacecolor='none', markersize=1, color="tab:orange", label="Previous")
        plt.plot(P_model[0], P_model[1], '+', markerfacecolor='none', markersize=1, color="tab:green", label="Model")
        for idx in range(P1.shape[1]):
            x1 = P1[0, idx]
            y1 = P1[1, idx]
            x2 = P2[0, idx]
            y2 = P2[1, idx]
            x3 = P_model[0, idx]
            y3 = P_model[1, idx]
            plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
            plt.plot([x2, x3], [y2, y3], 'r', linewidth=0.5, alpha=1)

            matched_points.append([x1, x2, y1, y2])

        plt.title("SVD on all proposed matches")
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig("%s%s%i%s" % (output_path, "/full_svd_landmark_matches_", i, ".pdf"))
        plt.close()

        print("SVD motion estimate (y, x, th):", pose_from_svd)

        # Motion estimate from RANSAC
        pose_estimates = get_pose_estimates_with_ransac(P1, P2, iterations=10)
        print(pose_estimates)
        best_model_index = get_best_ransac_motion_estimate_index(P1, P2, pose_estimates, figpath=figure_path)
        best_ransac_motion_estimate = pose_estimates[best_model_index]
        print("Best RANSAC motion estimate (y, x, th):", best_ransac_motion_estimate)


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running script...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    ransac_motion_estimation(params, radar_state_mono)


if __name__ == "__main__":
    main()

# Script to estimate R and theta from landmarks for imposing circular motion model
# python circular_motion_estimator.py --input_path "/workspace/data/landmark-distortion/ro_state_pb_developing/"
# --num_samples 80

import numpy as np
import matplotlib.pyplot as plt
import statistics
import traceback, sys, code
from pathlib import Path
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
import operator
import settings
import pdb
# from pyslam.metrics import TrajectoryMetrics
from pose_tools.pose_utils import *
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb
from get_rigid_body_motion import get_motion_estimate_from_svd
from R_and_theta_utilities import get_relative_range_and_bearing_from_x_and_y, get_theta_and_curvature_from_single_match
from kinematics import get_transform_by_r_and_theta

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


@dataclass
class CircularMotionEstimate:
    theta: float
    curvature: float
    range_1: float
    range_2: float
    bearing_1: float
    bearing_2: float


def circular_motion_estimation(params, radar_state_mono):
    figure_path = params.input_path + "figs_circular_motion_estimation/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    poses_from_full_match_set = []
    poses_from_circular_motion = []
    timestamps_from_ro_state = []

    for i in range(params.num_samples):
        # i = 3  # just for debugging because I know frame 3 is problematic
        pb_state, name_scan, _ = radar_state_mono[i]
        ro_state = get_ro_state_from_pb(pb_state)
        timestamps_from_ro_state.append(ro_state.timestamp)

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
        matched_points = []
        plt.figure(figsize=(10, 10))

        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]

            plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
            matched_points.append([x1, x2, y1, y2])

        circular_motion_estimates = []

        for tmp_idx in range(len(matched_points)):
            x1 = matched_points[tmp_idx][3]
            y1 = matched_points[tmp_idx][1]
            x2 = matched_points[tmp_idx][2]
            y2 = matched_points[tmp_idx][0]

            # if x1 == x2 and y1 == y2:
            #     print("\t\t\t*** x1 == x2 and y1 == y2 for idx:", tmp_idx)
            # else:
            r1, a1 = get_relative_range_and_bearing_from_x_and_y(relative_x=x1, relative_y=y1)
            r2, a2 = get_relative_range_and_bearing_from_x_and_y(relative_x=x2, relative_y=y2)
            theta, curvature = get_theta_and_curvature_from_single_match(d_1=r1, d_2=r2, phi_1=a1, phi_2=a2)
            circular_motion_estimates.append(
                CircularMotionEstimate(theta=theta, curvature=curvature, range_1=r1, range_2=r2, bearing_1=a1,
                                       bearing_2=a2))

        # sort circular motion estimates by theta value
        circular_motion_estimates.sort(key=operator.attrgetter('theta'))

        middle_cme = []
        middle_cme_idxs = []
        cm_poses = []

        for idx in range(len(circular_motion_estimates) // 4, 3 * len(circular_motion_estimates) // 4):
            middle_cme.append(circular_motion_estimates[idx])
            middle_cme_idxs.append(idx)

            cm_poses.append(get_transform_by_r_and_theta(1 / circular_motion_estimates[idx].curvature,
                                                         circular_motion_estimates[idx].theta))

        # dx_value = statistics.median([motions[0, 3] for motions in cm_poses])
        # dy_value = statistics.median([motions[1, 3] for motions in cm_poses])
        # dth_value = statistics.median([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])
        # poses_from_circular_motion.append([dx_value, dy_value, dth_value])
        #############################################################################

        # plt.figure(figsize=(10, 10))
        # plt.plot([cme.theta for cme in circular_motion_estimates], '.')
        # plt.plot(middle_cme_idxs, [cme.theta for cme in middle_cme], '.')
        # plt.plot(len(circular_motion_estimates) // 2,
        #          circular_motion_estimates[len(circular_motion_estimates) // 2].theta, 'r*')
        # plt.title("Theta values")
        # plt.grid()
        # plt.ylim(-0.05, 0.05)
        # plt.savefig(
        #     "%s%s%i%s" % (figure_path, "/debugging_thetas", i, ".pdf"))
        # plt.close()

        tmp_thetas = [cme.theta for cme in middle_cme]
        tmp_curvatures = [cme.curvature for cme in middle_cme]

        num_histogram_bins = 300
        H, xedges, yedges = np.histogram2d(tmp_thetas, tmp_curvatures, bins=num_histogram_bins)

        x, y = np.argwhere(H == H.max())[0]

        tmp_pose = get_transform_by_r_and_theta(1 / np.average(yedges[y:y + 2]), np.average(xedges[x:x + 2]))
        print("Histogram pose:", tmp_pose)

        dx_value = tmp_pose[0, 3]
        dy_value = tmp_pose[1, 3]
        dth_value = np.arctan2(tmp_pose[1, 0], tmp_pose[0, 0])
        poses_from_circular_motion.append([dx_value, dy_value, dth_value])

        # plt.figure(figsize=(10, 10))
        # plt.hist2d([cme.theta for cme in middle_cme],
        #            [cme.curvature for cme in middle_cme],
        #            num_histogram_bins, density=False)
        # # plt.plot([1 / cme.radius for cme in middle_cme], '.')
        # plt.title("Theta vs curvature values")
        # plt.xlabel("Theta (rad)")
        # plt.ylabel("Curvature")
        # plt.grid()
        # plt.savefig("%s%s%i%s" % (figure_path, "/debugging_2d_histogram_", i, ".pdf"))
        # plt.close()
        #
        # plt.figure(figsize=(10, 10))
        # plt.hist([cme.theta for cme in circular_motion_estimates], 100, density=False, facecolor='tab:blue')
        # plt.title("Theta values")
        # plt.grid()
        # plt.savefig("%s%s" % (figure_path, "/debugging_thetas_histogram.pdf"))
        # plt.close()
        #
        # plt.figure(figsize=(10, 10))
        # dx_values = [motions[0, 3] for motions in cm_poses]
        # print("dx median:", statistics.median(dx_values))
        # plt.hist(dx_values, 100, density=False, facecolor='tab:blue')
        # plt.title("dx values")
        # plt.grid()
        # plt.savefig("%s%s%i%s" % (figure_path, "/debugging_dx_values_", i, ".pdf"))
        # plt.close()

        # Motion estimate from running SVD on all the points
        pose_from_svd = get_motion_estimates_from_svd_on_full_matches(matched_points)
        poses_from_full_match_set.append(pose_from_svd)
        print("SVD motion estimate (x, y, th):", pose_from_svd)

    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_full_match_set,
                                      pose_source="full_matches",
                                      export_folder=params.input_path)
    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_circular_motion,
                                      pose_source="cm_matches",
                                      export_folder=params.input_path)


def get_motion_estimates_from_svd_on_full_matches(matched_points):
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
    return pose_from_svd


def plot_csv_things(params):
    print("Plotting pose estimate data...")

    figure_path = params.input_path + "figs_circular_motion_estimation/"
    output_path = Path(figure_path)
    # if output_path.exists() and output_path.is_dir():
    #     shutil.rmtree(output_path)
    # output_path.mkdir(parents=True)

    # Pose estimates from SVD on the full set of matches
    full_match_timestamps, full_match_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.input_path + "full_matches_poses.csv")
    svd_x = [float(item[0]) for item in full_match_x_y_th]
    svd_y = [float(item[1]) for item in full_match_x_y_th]
    svd_th = [float(item[2]) for item in full_match_x_y_th]

    # Pose estimates from inliers only
    cm_timestamps, cm_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.input_path + "cm_matches_poses.csv")
    cm_x = [float(item[0]) for item in cm_x_y_th]
    cm_y = [float(item[1]) for item in cm_x_y_th]
    cm_th = [float(item[2]) for item in cm_x_y_th]

    plt.figure(figsize=(15, 5))
    dim = params.num_samples
    # plt.xlim(0, 150)
    plt.grid()
    plt.plot(cm_x, '.-', label="cm_x")
    plt.plot(cm_y, '.-', label="cm_y")
    plt.plot(cm_th, '.-', label="cm_th")
    plt.plot(svd_x, '.-', label="svd_x")
    plt.plot(svd_y, '.-', label="svd_y")
    plt.plot(svd_th, '.-', label="svd_th")
    plt.title("Pose estimates: RO vs circular motion vs ground-truth")
    plt.xlabel("Index")
    plt.ylabel("units/sample")
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

    print("Running script...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state_17.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    circular_motion_estimation(params, radar_state_mono)
    # plot_csv_things(params)


if __name__ == "__main__":
    main()

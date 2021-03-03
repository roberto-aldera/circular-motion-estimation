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
from pyslam.metrics import TrajectoryMetrics
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

        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            matched_points.append([x1, x2, y1, y2])

        circular_motion_estimates = get_circular_motion_estimates_from_matches(matched_points)

        # Useful debugging plotting to see what's going on (while keeping this function neat and tidy)
        # debugging_plotting(figure_path, index=i, circular_motion_estimates=circular_motion_estimates)

        pose_from_circular_motion = get_median_dx_dy_dth_from_circular_motion_estimates(circular_motion_estimates)
        # poses_from_circular_motion.append([dx_value, dy_value, dth_value])
        poses_from_circular_motion.append(pose_from_circular_motion)
        print("Pose from circular motion:", pose_from_circular_motion)

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


def get_median_dx_dy_dth_from_circular_motion_estimates(circular_motion_estimates):
    # sort circular motion estimates by theta value
    circular_motion_estimates.sort(key=operator.attrgetter('theta'))

    middle_cme = []
    middle_cme_idxs = []
    cm_poses = []

    # Simple way: use the second and third quarter (middle bit) as a means of discarding the outliers
    for idx in range(len(circular_motion_estimates) // 4, 3 * len(circular_motion_estimates) // 4):
        middle_cme.append(circular_motion_estimates[idx])
        middle_cme_idxs.append(idx)
        radius = np.inf
        if circular_motion_estimates[idx].curvature != 0:
            radius = 1 / circular_motion_estimates[idx].curvature
        cm_poses.append(get_transform_by_r_and_theta(radius,
                                                     circular_motion_estimates[idx].theta))

    dx_value = statistics.median([motions[0, 3] for motions in cm_poses])
    dy_value = statistics.median([motions[1, 3] for motions in cm_poses])
    dth_value = statistics.median([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])

    print("Indices of medians for dx, dy, dtheta:")
    print(np.argsort([motions[0, 3] for motions in cm_poses])[len([motions[0, 3] for motions in cm_poses]) // 2])
    print(np.argsort([motions[1, 3] for motions in cm_poses])[len([motions[1, 3] for motions in cm_poses]) // 2])
    print(np.argsort([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])[
              len([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses]) // 2])
    return [dx_value, dy_value, dth_value]


def get_circular_motion_estimates_from_matches(matched_points):
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
    return circular_motion_estimates


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


def debugging_plotting(figure_path, index, circular_motion_estimates):
    # A staging area for some plotting
    plt.figure(figsize=(10, 10))
    theta_values = [estimates.theta for estimates in circular_motion_estimates]
    curvature_values = [estimates.curvature for estimates in circular_motion_estimates]
    # norm_thetas = [float(i) / max(theta_values) for i in theta_values]
    # norm_curvatures = [float(i) / max(curvature_values) for i in curvature_values]
    # plt.plot(norm_curvatures, norm_thetas, '.')
    plt.plot(curvature_values, theta_values, '.')
    plt.title("Theta vs curvature")
    plt.grid()
    plt.xlabel("Curvature")
    plt.ylabel("Theta")
    # plt.ylim(-1, 1)
    # plt.xlim(-1, 1)
    plt.savefig("%s%s%i%s" % (figure_path, "/debugging_curvature_theta_", index, ".pdf"))
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.plot(np.sort(curvature_values), 'r.', label="curvature")
    plt.plot(np.sort(theta_values), 'b.', label="theta")
    plt.title("Sorted curvature and theta values")
    plt.grid()
    plt.ylim(-1, 1)
    # plt.xlim(-0.0001, 0.0001)
    plt.legend()
    plt.savefig("%s%s%i%s" % (figure_path, "/debugging_", index, ".pdf"))
    plt.close()

    # Plot some Gaussians
    import scipy.stats as stats
    import math
    plt.figure(figsize=(10, 10))
    mu = np.mean(theta_values)
    variance = np.var(theta_values)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    # plt.plot(x, stats.norm.pdf(x, mu, sigma), label="theta")

    mu = np.mean(curvature_values)
    variance = np.var(curvature_values)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label="curvature")
    plt.grid()
    plt.legend()
    plt.savefig("%s%s%i%s" % (figure_path, "/gaussian_", index, ".pdf"))
    plt.close()


def get_metrics(params):
    # Some code to run KITTI metrics over poses, based on pyslam TrajectoryMetrics
    figure_path = params.input_path + "figs_circular_motion_estimation/error_metrics/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    gt_se3s, gt_timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    gt_se3s = gt_se3s[settings.K_RADAR_INDEX_OFFSET:]

    # Pose estimates from full matches
    full_matches_timestamps, full_matches_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.input_path + "full_matches_poses.csv")
    full_matches_se3s = get_raw_se3s_from_x_y_th(full_matches_x_y_th)

    # Pose estimates from inliers only
    cm_timestamps, cm_x_y_th = get_timestamps_and_x_y_th_from_csv(params.input_path + "cm_matches_poses.csv")
    cm_se3s = get_raw_se3s_from_x_y_th(cm_x_y_th)

    # Quick cropping hack *****
    # cropped_size = 2000
    # gt_se3s = gt_se3s[:cropped_size]
    # full_matches_se3s = full_matches_se3s[:cropped_size]
    # cm_se3s = cm_se3s[:cropped_size]
    # **************************************************

    relative_pose_index = settings.K_RADAR_INDEX_OFFSET + 1
    relative_pose_timestamp = gt_timestamps[relative_pose_index]

    # ensure timestamps are within a reasonable limit of each other (microseconds)
    assert (full_matches_timestamps[0] - relative_pose_timestamp) < 500
    assert (cm_timestamps[0] - relative_pose_timestamp) < 500

    # ANOTHER QUICK CHECK:
    ro_x, ro_y, ro_th = get_x_y_th_from_se3s(full_matches_se3s)
    gt_x, gt_y, gt_th = get_x_y_th_from_se3s(gt_se3s)

    plt.figure(figsize=(15, 10))
    dim = params.num_samples
    # plt.xlim(0, dim)
    plt.grid()
    plt.plot(ro_x, '.-', label="ro_x")
    plt.plot(ro_y, '.-', label="ro_y")
    plt.plot(ro_th, '.-', label="ro_th")
    plt.plot(gt_x[:dim], '.-', label="gt_x")
    plt.plot(gt_y[:dim], '.-', label="gt_y")
    plt.plot(gt_th[:dim], '.-', label="gt_th")
    plt.title("Pose estimates: RO vs ground-truth")
    plt.xlabel("Time (s)")
    plt.ylabel("units/s")
    plt.legend()
    plt.savefig("%s%s" % (output_path, "/odometry_comparison_check.png"))
    plt.close()
    # *****************************************************************

    # *****************************************************************
    # CORRECTION: making global poses from the relative poses
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_global_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    fm_global_se3s = [np.identity(4)]
    for i in range(1, len(full_matches_se3s)):
        fm_global_se3s.append(fm_global_se3s[i - 1] @ full_matches_se3s[i])
    full_matches_global_SE3s = get_se3s_from_raw_se3s(fm_global_se3s)

    cm_global_se3s = [np.identity(4)]
    for i in range(1, len(cm_se3s)):
        cm_global_se3s.append(cm_global_se3s[i - 1] @ cm_se3s[i])
    cm_global_SE3s = get_se3s_from_raw_se3s(cm_global_se3s)
    # *****************************************************************

    segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    # segment_lengths = [10, 20]
    # segment_lengths = [100, 200, 300, 400]

    tm_gt_fullmatches = TrajectoryMetrics(gt_global_SE3s, full_matches_global_SE3s)
    print_trajectory_metrics(tm_gt_fullmatches, segment_lengths, data_name="full match")

    tm_gt_cm = TrajectoryMetrics(gt_global_SE3s, cm_global_SE3s)
    print_trajectory_metrics(tm_gt_cm, segment_lengths, data_name="cm")

    # Visualiser experimenting
    from pyslam.visualizers import TrajectoryVisualizer
    visualiser = TrajectoryVisualizer({"full_matches": tm_gt_fullmatches, "cm": tm_gt_cm})
    visualiser.plot_cum_norm_err(outfile="/workspace/data/visualised_metrics_tmp/cumulative_norm_errors.pdf")
    # visualiser.plot_norm_err(outfile="/workspace/data/visualised_metrics_tmp/norm_errors.pdf")
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="/workspace/data/visualised_metrics_tmp/segment_errors.pdf")
    visualiser.plot_topdown(which_plane='yx',  # this is a custom flip to conform to MRG convention, instead of xy
                            outfile="/workspace/data/visualised_metrics_tmp/topdown.pdf")


def print_trajectory_metrics(tm_gt_est, segment_lengths, data_name="this"):
    print("\nTrajectory Metrics for", data_name, "set:")
    # print("endpoint_error:", tm_gt_est.endpoint_error(segment_lengths))
    # print("segment_errors:", tm_gt_est.segment_errors(segment_lengths))
    # print("traj_errors:", tm_gt_est.traj_errors())
    # print("rel_errors:", tm_gt_est.rel_errors())
    # print("error_norms:", tm_gt_est.error_norms())
    print("mean_err:", tm_gt_est.mean_err())
    # print("cum_err:", tm_gt_est.cum_err())
    print("rms_err:", tm_gt_est.rms_err())


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
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state_91.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    circular_motion_estimation(params, radar_state_mono)
    # plot_csv_things(params)
    # get_metrics(params)


if __name__ == "__main__":
    main()

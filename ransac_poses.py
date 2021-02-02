# Script for getting motion estimate by running ransac on the matches to reject outliers
# Runs SVD over inliers from the best RANSAC model compared to over all matches

# python ransac_poses.py --input_path "/workspace/data/landmark-distortion/ro_state_pb_developing/" --num_samples 80


import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
import pdb
from pyslam.metrics import TrajectoryMetrics
from pose_tools.pose_utils import *
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


def ransac_motion_estimation(params, radar_state_mono):
    figure_path = params.input_path + "figs_ransac_motion_estimation/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # se3s, timestamps = get_ground_truth_poses_from_csv(
    #     "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    # # print(se3s[0])

    # ro_se3s = []
    # ro_timestamps = []

    poses_from_full_match_set = []
    poses_from_inliers = []
    timestamps_from_ro_state = []

    for i in range(params.num_samples):
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
        pose_from_svd = [v[1], v[0], theta_R]
        poses_from_full_match_set.append(pose_from_svd)

        print("SVD motion estimate (x, y, th):", pose_from_svd)

        # Running SVD on all inliers from the best RANSAC model
        pose_estimates = get_pose_estimates_with_ransac(P1, P2, iterations=100)
        champion_inliers = get_all_inliers_from_best_ransac_motion_estimate(P1, P2, pose_estimates)

        P1_inliers = P1[:, champion_inliers]
        P2_inliers = P2[:, champion_inliers]
        v, theta_R = get_motion_estimate_from_svd(P1_inliers, P2_inliers, weights=np.ones(P1_inliers.shape[1]))
        pose_from_svd = [v[1], v[0], theta_R]
        poses_from_inliers.append(pose_from_svd)
        print("Pose from SVD on best inliers:", pose_from_svd)

    # x_vals_full_match_set = [pose[1] for pose in poses_from_full_match_set]
    # y_vals_full_match_set = [pose[0] for pose in poses_from_full_match_set]
    #
    # x_vals_inliers = [pose[1] for pose in poses_from_inliers]
    # y_vals_inliers = [pose[0] for pose in poses_from_inliers]

    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_full_match_set,
                                      pose_source="full_matches",
                                      export_folder=params.input_path)
    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_inliers,
                                      pose_source="inliers",
                                      export_folder=params.input_path)


def plot_all_sources(params):
    print("Plotting pose estimate data...")

    figure_path = params.input_path + "figs_ransac_motion_estimation/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    se3s, timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")

    # Pose estimates from SVD on the full set of matches
    full_match_timestamps, full_match_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.input_path + "full_matches_poses.csv")
    full_match_dx, full_match_dy, full_match_dth = get_x_y_th_velocities_from_x_y_th(full_match_x_y_th,
                                                                                     full_match_timestamps)
    start_time_offset = full_match_timestamps[0]
    full_match_time_seconds = [(x - start_time_offset) / 1e6 for x in full_match_timestamps[1:]]

    # Pose estimates from inliers only
    inlier_timestamps, inlier_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.input_path + "inliers_poses.csv")
    inlier_dx, inlier_dy, inlier_dth = get_x_y_th_velocities_from_x_y_th(inlier_x_y_th, inlier_timestamps)
    start_time_offset = inlier_timestamps[0]
    inlier_time_seconds = [(x - start_time_offset) / 1e6 for x in inlier_timestamps[1:]]

    # start_time_offset = ro_timestamps[0]
    # ro_time_seconds = [(x - start_time_offset) / 1e6 for x in ro_timestamps[1:]]
    # ro_x, ro_y, ro_th = get_x_y_th_velocities_from_poses(ro_se3s, ro_timestamps)

    gt_start_time_offset = timestamps[0]
    gt_time_seconds = [(x - gt_start_time_offset) / 1e6 for x in timestamps[1:]]
    gt_x, gt_y, gt_th = get_x_y_th_velocities_from_poses(se3s, timestamps)
    gt_x = gt_x[settings.K_RADAR_INDEX_OFFSET:]

    plt.figure(figsize=(15, 5))
    dim = params.num_samples
    plt.xlim(0, 150)
    plt.grid()
    # plt.plot(ro_time_seconds, ro_x, '.-', label="ro_x")
    # plt.plot(ro_time_seconds, ro_y, '.-', label="ro_y")
    # plt.plot(ro_time_seconds, ro_th, '.-', label="ro_th")
    plt.plot(inlier_time_seconds, inlier_dx, '.-', label="inlier_x")
    plt.plot(inlier_time_seconds, inlier_dy, '.-', label="inlier_y")
    plt.plot(inlier_time_seconds, inlier_dth, '.-', label="inlier_th")
    plt.plot(full_match_time_seconds, full_match_dx, '.-', label="full_x")
    plt.plot(full_match_time_seconds, full_match_dy, '.-', label="full_y")
    plt.plot(full_match_time_seconds, full_match_dth, '.-', label="full_th")
    plt.plot(gt_time_seconds[:dim], gt_x[:dim], '.-', label="gt_x")
    plt.plot(gt_time_seconds[:dim], gt_y[:dim], '.-', label="gt_y")
    plt.plot(gt_time_seconds[:dim], gt_th[:dim], '.-', label="gt_th")
    plt.title("Pose estimates: RO vs Inliers vs ground-truth")
    plt.xlabel("Time (s)")
    plt.ylabel("units/s")
    plt.legend()
    plt.savefig("%s%s" % (output_path, "/odometry_comparison.pdf"))
    plt.close()


def get_metrics(params):
    # Some code to run KITTI metrics over poses, based on pyslam TrajectoryMetrics
    figure_path = params.input_path + "figs_ransac_motion_estimation/error_metrics/"
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
    inlier_timestamps, inlier_x_y_th = get_timestamps_and_x_y_th_from_csv(params.input_path + "inliers_poses.csv")
    inlier_se3s = get_raw_se3s_from_x_y_th(inlier_x_y_th)

    # A quick plot just to sanity check this thing
    # plt.figure(figsize=(15, 5))
    # dim = params.num_samples
    # plt.xlim(0, dim)
    # plt.grid()
    # x_full = [float(sample[0]) for sample in full_matches_x_y_th]
    # x_inlier = [float(sample[0]) for sample in inlier_x_y_th]
    # x_gt = [se3.trans[0] for se3 in gt_se3s]
    #
    # plt.plot(x_full, '.-', label="full_x")
    # plt.plot(x_inlier, '.-', label="inlier_x")
    # plt.plot(x_gt, '.-', label="gt_x")
    # plt.title("Test")
    # plt.xlabel("Sample index")
    # plt.ylabel("units/sample")
    # plt.legend()
    # plt.savefig("%s%s" % (output_path, "/quick_odometry_comparison.pdf"))
    # plt.close()

    relative_pose_index = settings.K_RADAR_INDEX_OFFSET + 1
    relative_pose_timestamp = gt_timestamps[relative_pose_index]

    # ensure timestamps are within a reasonable limit of each other (microseconds)
    assert (full_matches_timestamps[0] - relative_pose_timestamp) < 500
    assert (inlier_timestamps[0] - relative_pose_timestamp) < 500

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

    inlier_global_se3s = [np.identity(4)]
    for i in range(1, len(inlier_se3s)):
        inlier_global_se3s.append(inlier_global_se3s[i - 1] @ inlier_se3s[i])
    inlier_global_SE3s = get_se3s_from_raw_se3s(inlier_global_se3s)
    # *****************************************************************

    segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    tm_gt_fullmatches = TrajectoryMetrics(gt_global_SE3s, full_matches_global_SE3s)
    print_trajectory_metrics(tm_gt_fullmatches, segment_lengths, data_name="full match")

    tm_gt_inliers = TrajectoryMetrics(gt_global_SE3s, inlier_global_SE3s)
    print_trajectory_metrics(tm_gt_inliers, segment_lengths, data_name="inlier")

    # Visualiser experimenting
    from pyslam.visualizers import TrajectoryVisualizer
    visualiser = TrajectoryVisualizer({"full_matches": tm_gt_fullmatches, "inliers": tm_gt_inliers})
    visualiser.plot_cum_norm_err(outfile="/workspace/data/visualised_metrics_tmp/cumulative_norm_errors.pdf")
    # visualiser.plot_norm_err(outfile="/workspace/data/visualised_metrics_tmp/norm_errors.pdf")
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="/workspace/data/visualised_metrics_tmp/segment_errors.pdf")
    visualiser.plot_topdown(which_plane='xy',
                            outfile="/workspace/data/visualised_metrics_tmp/topdown.pdf")


def plot_ground_traces(params):
    figure_path = params.input_path + "figs_ransac_motion_estimation/ground_traces/"
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
    inlier_timestamps, inlier_x_y_th = get_timestamps_and_x_y_th_from_csv(params.input_path + "inliers_poses.csv")
    inlier_se3s = get_raw_se3s_from_x_y_th(inlier_x_y_th)

    # Accumulate poses
    gt_x_position, gt_y_position = get_poses(gt_se3s)
    full_matches_x_position, full_matches_y_position = get_poses(full_matches_se3s)
    inlier_x_position, inlier_y_position = get_poses(inlier_se3s)

    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(gt_x_position, gt_y_position, '.', label="gt", color=settings.GT_COLOUR)
    plt.plot(full_matches_x_position, full_matches_y_position, '.', label="full_matches", color=settings.RO_COLOUR)
    plt.plot(inlier_x_position, inlier_y_position, '.', label="inliers", color=settings.AUX1_COLOUR)
    plt.title("Accumulated pose")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()
    plt.savefig("%s%s" % (output_path, "/all_poses_comparison.pdf"))
    plt.close()


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
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    # ransac_motion_estimation(params, radar_state_mono)
    # plot_all_sources(params)
    get_metrics(params)
    plot_ground_traces(params)


if __name__ == "__main__":
    main()

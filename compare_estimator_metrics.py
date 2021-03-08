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


def get_metrics(params):
    # Some code to run KITTI metrics over poses, based on pyslam TrajectoryMetrics
    gt_se3s, gt_timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    gt_se3s = gt_se3s[settings.K_RADAR_INDEX_OFFSET:]

    # Pose estimates from full matches
    full_matches_timestamps, full_matches_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.path + "full_matches_poses.csv")
    full_matches_se3s = get_raw_se3s_from_x_y_th(full_matches_x_y_th)

    # Aux 1 - using the ransac inliers only in the SVD estimate - cropping from 6900 to just the first 2000 frames
    aux0_timestamps, aux0_x_y_th = get_timestamps_and_x_y_th_from_csv(params.path + "6900_ransac_inliers_svd_poses.csv")
    aux0_timestamps, aux0_x_y_th = aux0_timestamps[:2000], aux0_x_y_th[:2000]
    aux0_se3s = get_raw_se3s_from_x_y_th(aux0_x_y_th)

    # Aux 1 - using the median for dx, dy, dth (between the first and third quartile of sorted theta values)
    aux1_timestamps, aux1_x_y_th = get_timestamps_and_x_y_th_from_csv(params.path + "2000_medians_iqr_poses.csv")
    aux1_se3s = get_raw_se3s_from_x_y_th(aux1_x_y_th)

    # Aux 2 - using the mean for dx, dy, dth (between the first and third quartile of sorted theta values)
    aux2_timestamps, aux2_x_y_th = get_timestamps_and_x_y_th_from_csv(params.path + "2000_means_iqr_poses.csv")
    aux2_se3s = get_raw_se3s_from_x_y_th(aux2_x_y_th)

    # Aux 3
    aux3_timestamps, aux3_x_y_th = get_timestamps_and_x_y_th_from_csv(params.path + "2000_double_iqr_poses_theta_25_75.csv")
    aux3_se3s = get_raw_se3s_from_x_y_th(aux3_x_y_th)

    # Aux 4
    aux4_timestamps, aux4_x_y_th = get_timestamps_and_x_y_th_from_csv(params.path + "2000_double_iqr_poses_theta_25_75_v2.csv")
    aux4_se3s = get_raw_se3s_from_x_y_th(aux4_x_y_th)

    # Aux 5
    aux5_timestamps, aux5_x_y_th = get_timestamps_and_x_y_th_from_csv(params.path + "2000_double_iqr_poses_theta_10_90.csv")
    aux5_se3s = get_raw_se3s_from_x_y_th(aux5_x_y_th)

    relative_pose_index = settings.K_RADAR_INDEX_OFFSET + 1
    relative_pose_timestamp = gt_timestamps[relative_pose_index]

    # ensure timestamps are within a reasonable limit of each other (microseconds)
    assert (full_matches_timestamps[0] - relative_pose_timestamp) < 500
    assert (aux0_timestamps[0] - relative_pose_timestamp) < 500
    assert (aux1_timestamps[0] - relative_pose_timestamp) < 500
    assert (aux2_timestamps[0] - relative_pose_timestamp) < 500
    assert (aux3_timestamps[0] - relative_pose_timestamp) < 500
    assert (aux4_timestamps[0] - relative_pose_timestamp) < 500
    assert (aux5_timestamps[0] - relative_pose_timestamp) < 500

    # making global poses from the relative poses
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_global_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    fm_global_se3s = [np.identity(4)]
    for i in range(1, len(full_matches_se3s)):
        fm_global_se3s.append(fm_global_se3s[i - 1] @ full_matches_se3s[i])
    full_matches_global_SE3s = get_se3s_from_raw_se3s(fm_global_se3s)

    aux0_global_se3s = [np.identity(4)]
    for i in range(1, len(aux0_se3s)):
        aux0_global_se3s.append(aux0_global_se3s[i - 1] @ aux0_se3s[i])
    aux0_global_SE3s = get_se3s_from_raw_se3s(aux0_global_se3s)

    aux1_global_se3s = [np.identity(4)]
    for i in range(1, len(aux1_se3s)):
        aux1_global_se3s.append(aux1_global_se3s[i - 1] @ aux1_se3s[i])
    aux1_global_SE3s = get_se3s_from_raw_se3s(aux1_global_se3s)

    aux2_global_se3s = [np.identity(4)]
    for i in range(1, len(aux2_se3s)):
        aux2_global_se3s.append(aux2_global_se3s[i - 1] @ aux2_se3s[i])
    aux2_global_SE3s = get_se3s_from_raw_se3s(aux2_global_se3s)

    aux3_global_se3s = [np.identity(4)]
    for i in range(1, len(aux3_se3s)):
        aux3_global_se3s.append(aux3_global_se3s[i - 1] @ aux3_se3s[i])
    aux3_global_SE3s = get_se3s_from_raw_se3s(aux3_global_se3s)

    aux4_global_se3s = [np.identity(4)]
    for i in range(1, len(aux4_se3s)):
        aux4_global_se3s.append(aux4_global_se3s[i - 1] @ aux4_se3s[i])
    aux4_global_SE3s = get_se3s_from_raw_se3s(aux4_global_se3s)

    aux5_global_se3s = [np.identity(4)]
    for i in range(1, len(aux4_se3s)):
        aux5_global_se3s.append(aux5_global_se3s[i - 1] @ aux5_se3s[i])
    aux5_global_SE3s = get_se3s_from_raw_se3s(aux5_global_se3s)

    segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    # segment_lengths = [10, 20]
    # segment_lengths = [100, 200, 300, 400]

    tm_gt_fullmatches = TrajectoryMetrics(gt_global_SE3s, full_matches_global_SE3s)
    print_trajectory_metrics(tm_gt_fullmatches, segment_lengths, data_name="full match")

    tm_gt_aux0 = TrajectoryMetrics(gt_global_SE3s, aux0_global_SE3s)
    print_trajectory_metrics(tm_gt_aux0, segment_lengths, data_name=settings.AUX0_NAME)

    tm_gt_aux1 = TrajectoryMetrics(gt_global_SE3s, aux1_global_SE3s)
    print_trajectory_metrics(tm_gt_aux1, segment_lengths, data_name=settings.AUX1_NAME)

    tm_gt_aux2 = TrajectoryMetrics(gt_global_SE3s, aux2_global_SE3s)
    print_trajectory_metrics(tm_gt_aux2, segment_lengths, data_name=settings.AUX2_NAME)

    tm_gt_aux3 = TrajectoryMetrics(gt_global_SE3s, aux3_global_SE3s)
    print_trajectory_metrics(tm_gt_aux3, segment_lengths, data_name=settings.AUX3_NAME)

    tm_gt_aux4 = TrajectoryMetrics(gt_global_SE3s, aux4_global_SE3s)
    print_trajectory_metrics(tm_gt_aux4, segment_lengths, data_name=settings.AUX4_NAME)

    tm_gt_aux5 = TrajectoryMetrics(gt_global_SE3s, aux5_global_SE3s)
    print_trajectory_metrics(tm_gt_aux5, segment_lengths, data_name=settings.AUX5_NAME)

    # Visualiser experimenting
    from pyslam.visualizers import TrajectoryVisualizer
    output_path_for_metrics = Path(params.path + "visualised_metrics")
    if output_path_for_metrics.exists() and output_path_for_metrics.is_dir():
        shutil.rmtree(output_path_for_metrics)
    output_path_for_metrics.mkdir(parents=True)

    visualiser = TrajectoryVisualizer(
        {"full_matches": tm_gt_fullmatches, settings.AUX0_NAME: tm_gt_aux0, settings.AUX1_NAME: tm_gt_aux1,
         settings.AUX2_NAME: tm_gt_aux2, settings.AUX3_NAME: tm_gt_aux3, settings.AUX4_NAME: tm_gt_aux4,
         settings.AUX5_NAME: tm_gt_aux5})
    visualiser.plot_cum_norm_err(outfile="%s%s" % (output_path_for_metrics, "/cumulative_norm_errors.pdf"))
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="%s%s" % (output_path_for_metrics, "/segment_errors.pdf"))
    visualiser.plot_topdown(which_plane='yx',  # this is a custom flip to conform to MRG convention, instead of xy
                            outfile="%s%s" % (output_path_for_metrics, "/topdown.pdf"))


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
    parser.add_argument('--path', type=str, default="",
                        help='Path to folder where inputs are and where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running script...")
    # python compare_estimator_metrics.py
    # --path "/workspace/data/landmark-distortion/ro_state_pb_developing/circular_motion_dev_metrics_comparison/" --num_samples 2000

    get_metrics(params)


if __name__ == "__main__":
    main()

# A script to check if the TrajectoryMetrics library is working as I expect it to, and to see if I can be sure I'm
# giving the correct inputs and expecting the correct outputs
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
from pyslam.metrics import TrajectoryMetrics
from pose_tools.pose_utils import *
import random
import pdb


def make_se3s():
    # Make some toy data, and then add some noise to make up the "estimate" of this toy data
    num_poses = 200
    gt_x = [1.0] * num_poses
    gt_y = [0.1] * num_poses
    gt_th = [0.01] * num_poses
    gt_x_y_th = [[gt_x[idx], gt_y[idx], gt_th[idx]] for idx in range(len(gt_x))]
    gt_se3s = get_raw_se3s_from_x_y_th(gt_x_y_th)

    x_errors = [random.uniform(-1, 1) for _ in range(len(gt_x))]
    y_errors = [random.uniform(-1, 1) for _ in range(len(gt_y))]
    th_errors = [random.uniform(-1, 1) for _ in range(len(gt_th))]

    x1_noise, y1_noise, th1_noise = 0.2, 0.1, 0.05
    est1_x = [gt_x[item] + x_errors[item] * x1_noise for item in range(len(gt_x))]
    est1_y = [gt_y[item] + y_errors[item] * y1_noise for item in range(len(gt_y))]
    est1_th = [gt_th[item] + th_errors[item] * th1_noise for item in range(len(gt_th))]
    est1_x_y_th = [[est1_x[idx], est1_y[idx], est1_th[idx]] for idx in range(len(est1_x))]
    est1_se3s = get_raw_se3s_from_x_y_th(est1_x_y_th)

    # Now make a second estimate trajectory, this time adding the exact same noise
    # as the first trajectory but in a slightly higher dose
    # Motivation for using the same noise -> if we sample new random noise,
    # there's a chance it'll introduce less error than the first trajectory's noise,
    # which defeats the whole purpose of having a second noisier estimate
    estimate_2_additional_error_factor = 1.2
    x2_noise, y2_noise, th2_noise = x1_noise * estimate_2_additional_error_factor, \
                                    y1_noise * estimate_2_additional_error_factor, \
                                    th1_noise * estimate_2_additional_error_factor

    est2_x = [gt_x[item] + x_errors[item] * x2_noise for item in range(len(gt_x))]
    est2_y = [gt_y[item] + y_errors[item] * y2_noise for item in range(len(gt_y))]
    est2_th = [gt_th[item] + th_errors[item] * th2_noise for item in range(len(gt_th))]
    est2_x_y_th = [[est2_x[idx], est2_y[idx], est2_th[idx]] for idx in range(len(est2_x))]
    est2_se3s = get_raw_se3s_from_x_y_th(est2_x_y_th)

    return gt_se3s, est1_se3s, est2_se3s


def get_metrics(gt_se3s, est1_se3s, est2_se3s):
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    est1_global_se3s = [np.identity(4)]
    for i in range(1, len(est1_se3s)):
        est1_global_se3s.append(est1_global_se3s[i - 1] @ est1_se3s[i])
    est1_SE3s = get_se3s_from_raw_se3s(est1_global_se3s)

    est2_global_se3s = [np.identity(4)]
    for i in range(1, len(est2_se3s)):
        est2_global_se3s.append(est2_global_se3s[i - 1] @ est2_se3s[i])
    est2_SE3s = get_se3s_from_raw_se3s(est2_global_se3s)

    segment_lengths = [1, 2, 3, 4]
    tm_gt_est1 = TrajectoryMetrics(gt_SE3s, est1_SE3s)
    tm_gt_est2 = TrajectoryMetrics(gt_SE3s, est2_SE3s)

    print("rms_err1:", tm_gt_est1.rms_err())
    print("rms_err2:", tm_gt_est2.rms_err())

    # print_trajectory_metrics(tm_gt_est1, segment_lengths)

    # Visualiser experimenting
    from pyslam.visualizers import TrajectoryVisualizer
    visualiser = TrajectoryVisualizer({"estimate_1": tm_gt_est1, "estimate_2": tm_gt_est2})
    visualiser.plot_cum_norm_err(
        outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/cumulative_norm_errors.pdf")
    visualiser.plot_norm_err(outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/norm_errors.pdf")
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/segment_errors.pdf")
    visualiser.plot_topdown(which_plane='xy',
                            outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/topdown.pdf")


def print_trajectory_metrics(tm_gt_est, segment_lengths):
    print("\nTrajectory Metrics for this set:")
    print("segment_errors:", tm_gt_est.segment_errors(segment_lengths))
    print("traj_errors:", tm_gt_est.traj_errors())
    print("rel_errors:", tm_gt_est.rel_errors())
    print("endpoint_error:", tm_gt_est.endpoint_error())
    print("error_norms:", tm_gt_est.error_norms())
    print("mean_err:", tm_gt_est.mean_err())
    print("cum_err:", tm_gt_est.cum_err())
    print("rms_err:", tm_gt_est.rms_err())


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--output_path', type=str, default="",
                        help='Path to folder to save outputs')
    parameters = parser.parse_args()

    print("Running script...")
    ground_truth_se3s, estimate1_se3s, estimate2_se3s = make_se3s()
    get_metrics(ground_truth_se3s, estimate1_se3s, estimate2_se3s)

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
    num_poses = 50
    gt_x = [1.0] * num_poses
    gt_y = [0.1] * num_poses
    gt_th = [0.05] * num_poses
    gt_x_y_th = [[gt_x[idx], gt_y[idx], gt_th[idx]] for idx in range(len(gt_x))]
    gt_se3s = get_raw_se3s_from_x_y_th(gt_x_y_th)

    x_noise, y_noise, th_noise = 0.1, 0.1, 0.05
    est_x = [gt_x[item] + random.uniform(-1, 1) * x_noise for item in range(len(gt_x))]
    est_y = [gt_y[item] + random.uniform(-1, 1) * y_noise for item in range(len(gt_y))]
    est_th = [gt_th[item] + random.uniform(-1, 1) * th_noise for item in range(len(gt_th))]
    est_x_y_th = [[est_x[idx], est_y[idx], est_th[idx]] for idx in range(len(est_x))]
    est_se3s = get_raw_se3s_from_x_y_th(est_x_y_th)

    return gt_se3s, est_se3s


def plot_cumulative_poses(gt_se3s, est_se3s, params):
    figure_path = params.output_path + "figs_test_the_metrics/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    gt_x_position, gt_y_position = get_poses(gt_se3s)
    est_x_position, est_y_position = get_poses(est_se3s)

    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(gt_x_position, gt_y_position, '.-', label="gt", color="tab:blue")
    plt.plot(est_x_position, est_y_position, '.-', label="estimate", color="tab:red")
    plt.title("Accumulated pose")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()
    plt.savefig("%s%s" % (output_path, "/all_poses_comparison.pdf"))
    plt.close()


def get_metrics(gt_se3s, est_se3s):
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    est_global_se3s = [np.identity(4)]
    for i in range(1, len(est_se3s)):
        est_global_se3s.append(est_global_se3s[i - 1] @ est_se3s[i])
    est_SE3s = get_se3s_from_raw_se3s(est_global_se3s)

    segment_lengths = [1, 2, 3, 4]
    tm_gt_est = TrajectoryMetrics(gt_SE3s, est_SE3s)

    print("\nTrajectory Metrics for this set:")
    print("segment_errors:", tm_gt_est.segment_errors(segment_lengths))
    print("traj_errors:", tm_gt_est.traj_errors())
    print("rel_errors:", tm_gt_est.rel_errors())
    print("endpoint_error:", tm_gt_est.endpoint_error())
    print("error_norms:", tm_gt_est.error_norms())
    print("mean_err:", tm_gt_est.mean_err())
    print("cum_err:", tm_gt_est.cum_err())
    print("rms_err:", tm_gt_est.rms_err())

    # Visualiser experimenting
    from pyslam.visualizers import TrajectoryVisualizer
    visualiser = TrajectoryVisualizer({"estimate": tm_gt_est})
    visualiser.plot_cum_norm_err(
        outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/cumulative_norm_errors.pdf")
    visualiser.plot_norm_err(outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/norm_errors.pdf")
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/segment_errors.pdf")
    visualiser.plot_topdown(which_plane='xy',
                            outfile="/workspace/data/landmark-distortion/figs_test_the_metrics/topdown.pdf")


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--output_path', type=str, default="",
                        help='Path to folder to save outputs')
    parameters = parser.parse_args()

    print("Running script...")
    ground_truth_se3s, estimate_se3s = make_se3s()
    plot_cumulative_poses(ground_truth_se3s, estimate_se3s, parameters)
    get_metrics(ground_truth_se3s, estimate_se3s)

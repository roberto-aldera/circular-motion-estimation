# Convert raw (unconstrained) ground truth poses into circular motion constrained poses by estimating a theta and
# curvature value from the raw SE3, before converting back to an SE3 that now conforms.

from tqdm import tqdm
from pose_tools.pose_utils import *
from kinematics import get_transform_by_translation_and_theta, get_transform_by_r_and_theta
import numpy as np
import matplotlib.pyplot as plt
import pdb
from argparse import ArgumentParser
import logging
from pyslam.metrics import TrajectoryMetrics
from pathlib import Path
import shutil
import settings
from dataclasses import dataclass

# create logger
logger = logging.getLogger('__name__')


@dataclass
class MotionEstimate:
    theta: float
    curvature: float
    dx: float
    dy: float
    dth: float


def get_cme_parameters(params, save_to_csv=True):
    se3s, timestamps = get_ground_truth_poses_from_csv(params.path + "/radar_odometry.csv")

    gt_x, gt_y, gt_th = get_x_y_th_from_se3s(se3s)
    motion_estimates = []
    se3s_from_cm_parameters = []

    num_iterations = min(params.num_samples, len(timestamps))
    print("Running for", num_iterations, "samples")

    # Try and find a theta and curvature value that gets as close to the pose as possible
    for idx in tqdm(range(num_iterations)):
        logger.debug(se3s[idx])

        se3_gt = se3s[idx]
        th_gt = np.arctan2(se3_gt[1, 0], se3_gt[0, 0])
        x_gt = se3_gt[0, 3]
        y_gt = se3_gt[1, 3]

        if th_gt != 0 and x_gt != 0:
            r_estimate = x_gt / np.sin(th_gt)
        else:
            r_estimate = np.inf
            th_gt = 0
        logger.debug(f'R, theta estimate: {r_estimate, th_gt}')

        se3_from_r_theta = get_transform_by_r_and_theta(r_estimate, th_gt)
        se3s_from_cm_parameters.append(se3_from_r_theta)
        x_est = se3_from_r_theta[0, 3]
        y_est = se3_from_r_theta[1, 3]
        th_est = np.arctan2(se3_from_r_theta[1, 0], se3_from_r_theta[0, 0])
        motion_estimates.append(MotionEstimate(theta=th_gt, curvature=1 / r_estimate, dx=x_est, dy=y_est, dth=th_est))
        logger.debug(f'GT: {x_gt}, {y_gt}, {th_gt}')
        logger.debug(f'Est: {x_est}, {y_est}, {th_est}')

    if save_to_csv:
        save_timestamps_and_cme_to_csv(timestamps[:num_iterations], motion_estimates, "gt", params.path)

    do_plotting = False
    if do_plotting:
        plt.figure(figsize=(15, 5))
        dim = num_iterations + 50
        plt.xlim(0, dim)
        plt.grid()
        plt.plot(gt_x, '+-', label="gt_x")
        plt.plot(gt_y, '+-', label="gt_y")
        plt.plot(gt_th, '+-', label="gt_th")
        plt.plot(cme_gt_x, '.-', label="cme_gt_x")
        plt.plot(cme_gt_y, '.-', label="cme_gt_y")
        plt.plot(cme_gt_th, '.-', label="cme_gt_th")
        plt.title("Pose estimates")
        plt.xlabel("Sample index")
        plt.ylabel("units/sample")
        plt.legend()
        plt.savefig("%s%s" % (params.path, "/pose_comparison.pdf"))
        plt.close()

    return se3s_from_cm_parameters


def save_timestamps_and_cme_to_csv(timestamps, motion_estimates, pose_source, export_folder):
    # Save poses with format: timestamp, theta, curvature, dx, dy, dth
    with open("%s%s%s" % (export_folder, pose_source, "_poses.csv"), 'w') as poses_file:
        wr = csv.writer(poses_file, delimiter=",")
        th_values = [item.dth for item in motion_estimates]
        curvature_values = [item.curvature for item in motion_estimates]
        x_values = [item.dx for item in motion_estimates]
        y_values = [item.dy for item in motion_estimates]
        th_values = [item.dth for item in motion_estimates]
        for idx in range(len(timestamps)):
            timestamp_and_motion_estimate = [timestamps[idx], th_values[idx], curvature_values[idx], x_values[idx],
                                             y_values[idx], th_values[idx]]
            wr.writerow(timestamp_and_motion_estimate)


def check_metrics(se3s_from_cm_parameters, params):
    gt_se3s, gt_timestamps = get_ground_truth_poses_from_csv(params.path + "/radar_odometry.csv")

    # making global poses from the relative poses
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_global_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    aux0_se3s = se3s_from_cm_parameters
    aux0_global_se3s = [np.identity(4)]
    for i in range(1, len(aux0_se3s)):
        aux0_global_se3s.append(aux0_global_se3s[i - 1] @ aux0_se3s[i])
    aux0_global_SE3s = get_se3s_from_raw_se3s(aux0_global_se3s)

    segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    tm_gt_cm = TrajectoryMetrics(gt_global_SE3s, aux0_global_SE3s)
    print_trajectory_metrics(tm_gt_cm, segment_lengths, data_name="CM-gt")

    # Visualise metrics
    from pyslam.visualizers import TrajectoryVisualizer
    output_path_for_metrics = Path(params.path + "visualised_metrics")
    if output_path_for_metrics.exists() and output_path_for_metrics.is_dir():
        shutil.rmtree(output_path_for_metrics)
    output_path_for_metrics.mkdir(parents=True)

    visualiser = TrajectoryVisualizer({"cm-gt": tm_gt_cm})
    visualiser.plot_cum_norm_err(outfile="%s%s" % (output_path_for_metrics, "/cumulative_norm_errors.pdf"))
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="%s%s" % (output_path_for_metrics, "/segment_errors.pdf"))
    visualiser.plot_topdown(which_plane='yx',  # this is a custom flip to conform to MRG convention, instead of xy
                            outfile="%s%s" % (output_path_for_metrics, "/topdown.pdf"), figsize=(10, 10))


def print_trajectory_metrics(tm_gt_est, segment_lengths, data_name="this"):
    print("\nTrajectory Metrics for", data_name, "set:")
    # print("endpoint_error:", tm_gt_est.endpoint_error(segment_lengths))
    # print("segment_errors:", tm_gt_est.segment_errors(segment_lengths))
    # print("traj_errors:", tm_gt_est.traj_errors())
    # print("rel_errors:", tm_gt_est.rel_errors())
    # print("error_norms:", tm_gt_est.error_norms())
    print("average segment_error:", np.mean(tm_gt_est.segment_errors(segment_lengths, rot_unit='deg')[1], axis=0)[1:])
    print("mean_err:", tm_gt_est.mean_err())
    # print("cum_err:", tm_gt_est.cum_err())
    print("rms_err:", tm_gt_est.rms_err())


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--path', type=str, default=settings.RO_STATE_PATH,
                        help='Path to folder containing required inputs')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Logging level')
    params = parser.parse_args()

    logging_level = logging.DEBUG if params.verbose > 0 else logging.INFO
    logger.setLevel(logging_level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logger.info("Running script...")

    # --input_path /workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv
    # --output_path /workspace/data/landmark-distortion/cme_ground_truth/
    # --num_samples 6900
    se3s_from_cm_parameters = get_cme_parameters(params, save_to_csv=True)
    # check_metrics(se3s_from_cm_parameters, params)

    logger.info("Finished.")

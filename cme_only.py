# Script to estimate R and theta from landmarks for imposing circular motion model
# python cme_only.py --input_path "/workspace/data/landmark-distortion/ro_state_pb_developing/ro_state_files/"
# --output_path "/workspace/data/landmark-distortion/ro_state_pb_developing/circular_motion_dev/" --num_samples 1

from tqdm import tqdm
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
import time
import csv
import pdb
import logging
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

# create logger
logger = logging.getLogger('__name__')


@dataclass
class CircularMotionEstimate:
    theta: float
    curvature: float
    range_1: float
    range_2: float
    bearing_1: float
    bearing_2: float


def circular_motion_estimation(params, radar_state_mono, results_path):
    poses_from_circular_motion = []
    poses_from_circular_motion_SVD = []
    timestamps_from_ro_state = []
    num_iterations = min(params.num_samples, len(radar_state_mono))
    print("Running for", num_iterations, "samples")

    for i in tqdm(range(num_iterations)):
        pb_state, name_scan, _ = radar_state_mono[i + 0]
        ro_state = get_ro_state_from_pb(pb_state)
        timestamps_from_ro_state.append(ro_state.timestamp)

        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        primary_landmarks = np.c_[
            primary_landmarks, np.ones(len(primary_landmarks))]  # so that se3 multiplication works

        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))

        logger.debug(f'Size of primary landmarks {len(primary_landmarks)}')
        logger.debug(f'Size of secondary landmarks: {len(secondary_landmarks)}')

        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        matches_to_plot = selected_matches.astype(int)

        logger.debug(f'Processing index: {i}')
        matched_points = []

        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            matched_points.append([x1, x2, y1, y2])

        circular_motion_estimates = get_circular_motion_estimates_from_matches(matched_points)

        # Theta plotting for figure exports
        theta_plotting(circular_motion_estimates, results_path)

        # Get pose using all CME-selected points and the SVD
        pose_from_circular_motion_SVD = get_svd_pose_from_circular_motion_estimates(matched_points,
                                                                                    circular_motion_estimates)
        poses_from_circular_motion_SVD.append(pose_from_circular_motion_SVD)

        # Useful debugging plotting to see what's going on (while keeping this function neat and tidy)
        # debugging_plotting(figure_path, index=i, circular_motion_estimates=circular_motion_estimates)

        pose_from_circular_motion = get_dx_dy_dth_from_circular_motion_estimates(circular_motion_estimates)
        poses_from_circular_motion.append(pose_from_circular_motion)
        logger.debug(f'Pose from circular motion: {pose_from_circular_motion}')

    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_circular_motion_SVD,
                                      pose_source="cm_matches_svd",
                                      export_folder=results_path)
    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_circular_motion,
                                      pose_source="cm_matches",
                                      export_folder=results_path)


def get_dx_dy_dth_from_circular_motion_estimates(circular_motion_estimates):
    cm_poses = []
    chosen_indices = []
    thetas = [cme.theta for cme in circular_motion_estimates]
    # sd_theta = np.std(thetas)
    # lower_theta_bound = np.mean(thetas) - sd_theta
    # upper_theta_bound = np.mean(thetas) + sd_theta
    percentile_start, percentile_end = 35, 65
    q1_theta, q3_theta = np.percentile(thetas, percentile_start), np.percentile(thetas, percentile_end)
    logger.debug(f'Q1 and Q3 for theta: {q1_theta}, {q3_theta}')

    # for i in range(len(circular_motion_estimates)):
    #     if (circular_motion_estimates[i].theta >= lower_theta_bound) and (
    #             circular_motion_estimates[i].theta <= upper_theta_bound):
    #         chosen_indices.append(i)
    for i in range(len(circular_motion_estimates)):
        if (circular_motion_estimates[i].theta >= q1_theta) and (
                circular_motion_estimates[i].theta <= q3_theta):
            chosen_indices.append(i)
    logger.debug(f'Using {len(chosen_indices)} out of {len(circular_motion_estimates)} circular motion estimates.')

    for idx in chosen_indices:
        radius = np.inf
        if circular_motion_estimates[idx].curvature != 0:
            radius = 1 / circular_motion_estimates[idx].curvature
        cm_poses.append(get_transform_by_r_and_theta(radius,
                                                     circular_motion_estimates[idx].theta))
    dx_value = statistics.mean([motions[0, 3] for motions in cm_poses])
    dy_value = statistics.mean([motions[1, 3] for motions in cm_poses])
    dth_value = statistics.mean([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])

    return [dx_value, dy_value, dth_value]


def get_svd_pose_from_circular_motion_estimates(matched_points, circular_motion_estimates):
    P1 = []
    P2 = []
    chosen_indices = []
    thetas = [cme.theta for cme in circular_motion_estimates]

    percentile_start, percentile_end = 35, 65
    q1_theta, q3_theta = np.percentile(thetas, percentile_start), np.percentile(thetas, percentile_end)
    logger.debug(f'Q1 and Q3 for theta: {q1_theta}, {q3_theta}')

    for i in range(len(circular_motion_estimates)):
        if (circular_motion_estimates[i].theta >= q1_theta) and (
                circular_motion_estimates[i].theta <= q3_theta):
            chosen_indices.append(i)
            P1.append([matched_points[i][0], matched_points[i][2]])
            P2.append([matched_points[i][1], matched_points[i][3]])

    P1 = np.transpose(P1)
    P2 = np.transpose(P2)
    v, theta_R = get_motion_estimate_from_svd(P1, P2, weights=np.ones(P1.shape[1]))
    return [v[1], v[0], -theta_R]


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


def theta_plotting(circular_motion_estimates, results_path):
    chosen_indices_35_65 = []
    thetas = [cme.theta for cme in circular_motion_estimates]
    sorted_thetas = np.sort(thetas)
    sample_indices = np.arange(0, len(sorted_thetas))

    q1_theta, q3_theta = np.percentile(thetas, 35), np.percentile(thetas, 65)
    for i in range(len(circular_motion_estimates)):
        if (sorted_thetas[i] >= q1_theta) and (sorted_thetas[i] <= q3_theta):
            chosen_indices_35_65.append(i)

    inner_thetas_35_65 = np.array(sorted_thetas)
    notIndex = np.array([i for i in sample_indices if i not in chosen_indices_35_65])
    inner_thetas_35_65[notIndex] = np.nan

    chosen_indices_1_std_dev = []
    mean_theta = np.mean(thetas)
    std_dev_theta = np.std(thetas)
    upper_bound = mean_theta + std_dev_theta
    lower_bound = mean_theta - std_dev_theta
    for i in range(len(circular_motion_estimates)):
        if (sorted_thetas[i] >= lower_bound) and (sorted_thetas[i] <= upper_bound):
            chosen_indices_1_std_dev.append(i)

    inner_thetas_1_std_dev = np.array(sorted_thetas)
    notIndex = np.array([i for i in sample_indices if i not in chosen_indices_1_std_dev])
    inner_thetas_1_std_dev[notIndex] = np.nan

    # Plot thetas here for generating figures
    font_size = 16
    marker_size = 3
    line_width = 3
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.figure(figsize=(10, 5))
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid()

    plt.plot(sample_indices, sorted_thetas, 'o', color="tab:red", ms=marker_size, label="Sorted thetas")
    # plt.plot(sample_indices, inner_thetas_1_std_dev, ',', color="tab:green", ms=marker_size, label="Std Dev thetas")
    # plt.plot(sample_indices, inner_thetas_35_65, 'x', color="tab:blue", ms=marker_size, label="IQR thetas")
    plt.vlines([np.min(chosen_indices_1_std_dev), np.max(chosen_indices_1_std_dev)], ymin=-1, ymax=1,
               linestyles='dashed', color="tab:green", lw=line_width, label="1σ limits")
    plt.vlines([np.min(chosen_indices_35_65), np.max(chosen_indices_35_65)], ymin=-1, ymax=1, linestyles='dashed',
               color="tab:blue", lw=line_width, label="Quantile limits")
    plt.title("Subset selection based on θ-values", fontsize=font_size)
    plt.xlabel("Sample index", fontsize=font_size)
    plt.ylabel("Theta (rad)", fontsize=font_size)
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.tight_layout()
    figure_path = "%s%s" % (results_path, "sorted_thetas.pdf")
    plt.savefig(figure_path)
    plt.close()
    print("Saved figure to:", figure_path)
    pdb.set_trace()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default=settings.RO_STATE_PATH,
                        help='Path to folder containing required inputs')
    parser.add_argument('--output_path', type=str, default=settings.POSE_OUTPUT_PATH,
                        help='Path to folder where outputs will be saved')
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

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    logger.info(f'Number of indices in this radar odometry state monolithic: {len(radar_state_mono)}')

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    results_path = settings.POSE_OUTPUT_PATH + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    circular_motion_estimation(params, radar_state_mono, results_path)


if __name__ == "__main__":
    main()

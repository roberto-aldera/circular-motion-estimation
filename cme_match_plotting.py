# Script to estimate R and theta from landmarks for imposing circular motion model
# python cme_match_plotting.py --input_path /workspace/data/ro-state-files/radar_oxford_10k/2019-01-10-14-50-05/ --num_samples 1

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
from pose_tools.pose_utils import *

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud

# create logger
logger = logging.getLogger('__name__')

plt.rc('text', usetex=False)
plt.rc('font', family='serif')


@dataclass
class CircularMotionEstimate:
    theta: float
    curvature: float
    range_1: float
    range_2: float
    bearing_1: float
    bearing_2: float


def make_landmark_deltas_figure(params, radar_state_mono, results_path):
    se3s, timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    global_pose = np.eye(4)

    k_start_index_from_odometry = 370  # 260

    for i in range(params.num_samples):
        plt.figure(figsize=(8, 8))
        dim = 60
        plt.xlim(-dim, dim)
        plt.ylim(-dim, dim)
        pb_state, name_scan, _ = radar_state_mono[i + k_start_index_from_odometry]
        ro_state = get_ro_state_from_pb(pb_state)
        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        primary_landmarks = np.c_[
            primary_landmarks, np.ones(len(primary_landmarks))]  # so that se3 multiplication works

        relative_pose_index = i + k_start_index_from_odometry + 1
        relative_pose_timestamp = timestamps[relative_pose_index]
        print("Relative pose:", se3s[relative_pose_index])

        # ensure timestamps are within a reasonable limit of each other (microseconds)
        assert (ro_state.timestamp - relative_pose_timestamp) < 500

        primary_landmarks = np.transpose(global_pose @ np.transpose(primary_landmarks))

        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))

        print("Size of primary landmarks:", len(primary_landmarks))
        print("Size of secondary landmarks:", len(secondary_landmarks))

        matches_to_plot = selected_matches.astype(int)

        matched_points = []
        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            matched_points.append([x1, x2, y1, y2])

        circular_motion_estimates = get_circular_motion_estimates_from_matches(matched_points)
        pose_from_circular_motion, chosen_indices = get_dx_dy_dth_from_circular_motion_estimates(
            circular_motion_estimates)

        print("Processing index: ", i)
        # Generate some random indices to subsample matches (figure is too dense otherwise for illustration purposes)
        # num_subsamples = 300
        # sample_indices = np.random.randint(low=0, high=len(matches_to_plot), size=num_subsamples)

        # plot x and y swapped around so that robot is moving forward as upward direction
        for match_idx in range(len(matches_to_plot)):  # chosen_indices:  # sample_indices:
            # if match_idx in sample_indices:
            # if match_idx in chosen_indices:
            # for match_idx in chosen_indices:  # sample_indices:
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]

            do_red_green_plots = False
            if do_red_green_plots:
                if match_idx in chosen_indices:
                    plt.plot([x1, x2], [y1, y2], 'g', linewidth=2.0)  # , alpha=normalised_match_weight[match_idx])
                    # continue
                else:
                    plt.plot(x1, y1, 'o', markersize=5, markerfacecolor='none', color="tab:blue")
                    plt.plot(x2, y2, 'o', markersize=5, markerfacecolor='none', color="tab:orange")
                    plt.plot([x1, x2], [y1, y2], 'r', linewidth=2.0)  # , alpha=normalised_match_weight[match_idx])
            else:
                plt.plot(x1, y1, 'o', markersize=5, markerfacecolor='none', color="tab:blue")
                plt.plot(x2, y2, 'o', markersize=5, markerfacecolor='none', color="tab:orange")
                plt.plot([x1, x2], [y1, y2], 'k', linewidth=2.0)

        robot_element, = plt.plot(0, 0, '^', markerfacecolor="tab:green", markeredgecolor="green", markersize=10,
                                  label="Robot")
        p1_element, = plt.plot([], [], ".", markersize=10, color="tab:blue", label="Primary landmarks")
        p2_element, = plt.plot([], [], ".", markersize=10, color="tab:orange", label="Secondary landmarks")
        matches_element, = plt.plot([], [], color="k", linewidth=2.0, label="Matches")
        font_size = 16

        plot_with_labels = True
        if plot_with_labels:
            plt.legend(handles=[p1_element, p2_element, matches_element, robot_element], fontsize=font_size)
            plt.title("Correspondences between two sequential landmark sets", fontsize=font_size)
            plt.xlabel("Y-Position (m)", fontsize=font_size)
            plt.ylabel("X-Position (m)", fontsize=font_size)
        plt.grid()
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("%s%s%i%s" % (results_path, "/matched_landmarks_subset_", i, ".pdf"))
        plt.close()


# Similar to function make_landmark_deltas_figure but with an added inset just to clarify a figure in the paper
def make_landmark_deltas_figure_with_inset(params, radar_state_mono, results_path):
    se3s, timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    global_pose = np.eye(4)

    k_start_index_from_odometry = 370

    for i in range(params.num_samples):
        plt.figure(figsize=(8, 8))
        ax = plt.axes()
        dim = 60
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)

        # Zooming bit starts here
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        x_start = -10
        x_end = -4
        y_start = 22
        y_end = 28

        # Make the zoom-in plot:
        axins = zoomed_inset_axes(ax, 5, loc='center right')
        axins.set_xlim(x_start, x_end)
        axins.set_ylim(y_start, y_end)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

        pb_state, name_scan, _ = radar_state_mono[i + k_start_index_from_odometry]
        ro_state = get_ro_state_from_pb(pb_state)
        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        primary_landmarks = np.c_[
            primary_landmarks, np.ones(len(primary_landmarks))]  # so that se3 multiplication works

        relative_pose_index = i + k_start_index_from_odometry + 1
        relative_pose_timestamp = timestamps[relative_pose_index]
        print("Relative pose:", se3s[relative_pose_index])

        # ensure timestamps are within a reasonable limit of each other (microseconds)
        assert (ro_state.timestamp - relative_pose_timestamp) < 500

        primary_landmarks = np.transpose(global_pose @ np.transpose(primary_landmarks))

        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))

        print("Size of primary landmarks:", len(primary_landmarks))
        print("Size of secondary landmarks:", len(secondary_landmarks))

        matches_to_plot = selected_matches.astype(int)

        matched_points = []
        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            matched_points.append([x1, x2, y1, y2])

        circular_motion_estimates = get_circular_motion_estimates_from_matches(matched_points)
        pose_from_circular_motion, chosen_indices = get_dx_dy_dth_from_circular_motion_estimates(
            circular_motion_estimates)

        print("Processing index: ", i)
        # Generate some random indices to subsample matches (figure is too dense otherwise for illustration purposes)
        # num_subsamples = 300
        # sample_indices = np.random.randint(low=0, high=len(matches_to_plot), size=num_subsamples)

        # plot x and y swapped around so that robot is moving forward as upward direction
        for match_idx in range(len(matches_to_plot)):  # chosen_indices:  # sample_indices:
            # if match_idx in sample_indices:
            # if match_idx in chosen_indices:
            # for match_idx in chosen_indices:  # sample_indices:
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]

            ax.plot(x1, y1, 'o', markersize=5, markerfacecolor='none', color="tab:blue")
            ax.plot(x2, y2, 'o', markersize=5, markerfacecolor='none', color="tab:orange")
            ax.plot([x1, x2], [y1, y2], 'k', linewidth=2.0)

            axins.plot(x1, y1, 'o', markersize=5, markerfacecolor='none', color="tab:blue")
            axins.plot(x2, y2, 'o', markersize=5, markerfacecolor='none', color="tab:orange")
            axins.plot([x1, x2], [y1, y2], 'k', linewidth=2.0)

        robot_element, = ax.plot(0, 0, '^', markerfacecolor="tab:green", markeredgecolor="green", markersize=10,
                                 label="Robot")
        p1_element, = ax.plot([], [], ".", markersize=10, color="tab:blue", label="Primary landmarks")
        p2_element, = ax.plot([], [], ".", markersize=10, color="tab:orange", label="Secondary landmarks")
        matches_element, = ax.plot([], [], color="k", linewidth=2.0, label="Matches")

        font_size = 16
        ax.legend(handles=[p1_element, p2_element, matches_element, robot_element], fontsize=font_size)
        ax.set_title("Correspondences between two sequential landmark sets", fontsize=font_size)
        ax.set_xlabel("Y-Position (m)", fontsize=font_size)
        ax.set_ylabel("X-Position (m)", fontsize=font_size)
        ax.grid()
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig("%s%s%i%s" % (results_path, "/matched_landmarks_subset_", i, ".pdf"))
        plt.close()


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

    return [dx_value, dy_value, dth_value], chosen_indices


def get_svd_pose_from_circular_motion_estimates(matched_points, circular_motion_estimates):
    P1 = []
    P2 = []
    chosen_indices = []
    thetas = [cme.theta for cme in circular_motion_estimates]

    percentile_start, percentile_end = 25, 75
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


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default=settings.RO_STATE_PATH,
                        help='Path to folder containing required inputs')
    parser.add_argument('--num_samples', type=int, default=1,
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
    results_path = params.input_path + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    print("Results to be stored in:", results_path)

    # circular_motion_estimation(params, radar_state_mono, results_path)
    make_landmark_deltas_figure_with_inset(params, radar_state_mono, results_path)


if __name__ == "__main__":
    main()

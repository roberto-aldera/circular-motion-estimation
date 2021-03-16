# Script to estimate R and theta from landmarks for imposing circular motion model
# python circular_motion_estimator.py --input_path "/workspace/data/landmark-distortion/ro_state_pb_developing/ro_state_files/"
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
import pdb
import logging
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


def circular_motion_estimation(params, radar_state_mono):
    figure_path = params.output_path + "figs_circular_motion_estimation/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    poses_from_full_match_set = []
    poses_from_circular_motion = []
    timestamps_from_ro_state = []

    for i in tqdm(range(params.num_samples)):
        pb_state, name_scan, _ = radar_state_mono[i]
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

        # Useful debugging plotting to see what's going on (while keeping this function neat and tidy)
        # debugging_plotting(figure_path, index=i, circular_motion_estimates=circular_motion_estimates)
        # plot_sorted_values(figure_path, index=i, circular_motion_estimates=circular_motion_estimates)
        # plot_2d_kde_values(figure_path, index=i, circular_motion_estimates=circular_motion_estimates)

        ############################################################################################
        pose_from_circular_motion = get_dx_dy_dth_from_circular_motion_estimates_kde_2d(circular_motion_estimates)
        # pose_from_circular_motion = get_dx_dy_dth_from_circular_motion_estimates_double_iqr(circular_motion_estimates)
        # pose_from_circular_motion = do_experimental_local_outlier_factor(circular_motion_estimates, idx=i)
        # pose_from_circular_motion = get_median_dx_dy_dth_from_circular_motion_estimates(circular_motion_estimates)
        # pose_from_circular_motion = get_experimental_dx_dy_dth_from_circular_motion_estimates(
        #     circular_motion_estimates)
        poses_from_circular_motion.append(pose_from_circular_motion)
        logger.debug(f'Pose from circular motion: {pose_from_circular_motion}')

        # Motion estimate from running SVD on all the points
        pose_from_svd = get_motion_estimates_from_svd_on_full_matches(matched_points)
        poses_from_full_match_set.append(pose_from_svd)
        logger.debug(f'SVD motion estimate (x, y, th) {pose_from_svd}')

    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_full_match_set,
                                      pose_source="full_matches",
                                      export_folder=params.output_path)
    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_circular_motion,
                                      pose_source="cm_matches",
                                      export_folder=params.output_path)


def get_dx_dy_dth_from_circular_motion_estimates_kde_2d(circular_motion_estimates):
    # Need to drop circular motion estimates where curvature values could be np.inf:
    validated_circular_motion_estimates = []
    for cme in circular_motion_estimates:
        if cme.curvature != np.inf:
            validated_circular_motion_estimates.append(cme)
    from sklearn.neighbors import KernelDensity
    thetas = np.array([cme.theta for cme in validated_circular_motion_estimates])
    curvatures = np.array([cme.curvature for cme in validated_circular_motion_estimates])

    # m1 = thetas
    # m2 = curvatures
    # x, y = m1 + m2, m1 - m2
    x, y = thetas, curvatures
    value_window_dimension = 0.5
    num_bins = 100
    cell_size = 2 * value_window_dimension / num_bins
    xx, yy, zz = kde2D(x, y, bandwidth=0.05, dim=value_window_dimension)

    # Find "hottest" cell in heatmap
    max_zz_indices = np.unravel_index(zz.argmax(), zz.shape)
    # Get theta and curvature thresholds from the cell's limits - this is a center-valued cell (so divide by 2)
    # Perhaps later it would be worth taking nearby cells too if necessary - divide by 1 to widen things for now
    width_constant = 1
    theta_min, theta_max = xx[max_zz_indices[0]][0] - (cell_size * width_constant), xx[max_zz_indices[0]][0] + (
                cell_size * width_constant)
    curvature_min, curvature_max = yy[0][max_zz_indices[1]] - cell_size * width_constant, yy[0][
        max_zz_indices[1]] + cell_size * width_constant

    # Select indices from circular match estimates that correspond to matches within KDE bounds
    # Keep all indices where theta is between these two elements (including them)
    selected_indices_based_on_theta = []
    middle_thetas = []
    for index in range(len(thetas)):
        theta = validated_circular_motion_estimates[index].theta
        if (theta >= theta_min) and (theta <= theta_max):
            middle_thetas.append(theta)
            selected_indices_based_on_theta.append(index)
    logger.debug(f'Thetas within the specified range: {len(middle_thetas)} of {len(thetas)}')

    # Keep all indices where curvature is between these two elements (including them)
    selected_indices_based_on_curvature = []
    middle_curvatures = []
    for index in range(len(curvatures)):
        curvature = validated_circular_motion_estimates[index].curvature
        if (curvature >= curvature_min) and (curvature <= curvature_max):
            middle_curvatures.append(curvature)
            selected_indices_based_on_curvature.append(index)
    logger.debug(f'Curvatures within the specified range: {len(middle_curvatures)} of {len(curvatures)}')

    # Find indices that are common between both middle ranges
    common_indices = list(set(selected_indices_based_on_theta).intersection(selected_indices_based_on_curvature))
    logger.debug(
        f'Indices common to both theta and curvature ranges: {len(common_indices)} of {len(circular_motion_estimates)}')

    cm_poses = []
    # for index in selected_indices_based_on_theta:
    # for index in selected_indices_based_on_curvature:
    for index in common_indices:
        radius = np.inf
        if validated_circular_motion_estimates[index].curvature != 0:
            radius = 1 / validated_circular_motion_estimates[index].curvature
        cm_poses.append(get_transform_by_r_and_theta(radius,
                                                     validated_circular_motion_estimates[index].theta))

    dx_value = statistics.mean([motions[0, 3] for motions in cm_poses])
    dy_value = statistics.mean([motions[1, 3] for motions in cm_poses])
    dth_value = statistics.mean([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])

    return [dx_value, dy_value, dth_value]


def get_dx_dy_dth_from_circular_motion_estimates_kde_1d(circular_motion_estimates):
    # Need to drop circular motion estimates where curvature values could be np.inf:
    validated_circular_motion_estimates = []
    for cme in circular_motion_estimates:
        if cme.curvature != np.inf:
            validated_circular_motion_estimates.append(cme)
    bw = 0.01
    from sklearn.neighbors import KernelDensity
    thetas = np.array([cme.theta for cme in validated_circular_motion_estimates])[:, np.newaxis]
    curvatures = np.array([cme.curvature for cme in validated_circular_motion_estimates])[:, np.newaxis]
    kde_thetas = KernelDensity(kernel='gaussian', bandwidth=bw).fit(thetas)
    kde_curvatures = KernelDensity(kernel='gaussian', bandwidth=bw).fit(curvatures)

    x_dim = 0.05
    x = np.linspace(-x_dim, x_dim, 1001)[:, np.newaxis]
    theta_density = np.exp(kde_thetas.score_samples(x))
    curvature_density = np.exp(kde_curvatures.score_samples(x))

    # Get max for theta and curvature from KDEs:
    best_theta = x[int(np.argmax(np.exp(kde_thetas.score_samples(x))))]
    best_curvature = x[int(np.argmax(np.exp(kde_curvatures.score_samples(x))))]
    # print("Best theta:", best_theta)
    # print("Best curvature:", best_curvature)

    radius = np.inf
    if best_curvature != 0:
        radius = 1 / best_curvature

    cm_pose = get_transform_by_r_and_theta(radius, best_theta)

    dx_value = cm_pose[0, 3]
    dy_value = cm_pose[1, 3]
    dth_value = np.arctan2(cm_pose[1, 0], cm_pose[0, 0])
    # pdb.set_trace()

    return [dx_value, dy_value, dth_value]


def get_dx_dy_dth_from_circular_motion_estimates_double_iqr(circular_motion_estimates):
    thetas = [cme.theta for cme in circular_motion_estimates]
    curvatures = [cme.curvature for cme in circular_motion_estimates]

    # get Q1 and Q3 element from thetas and curvatures
    percentile_start, percentile_end = 25, 75
    q1_theta, q3_theta = np.percentile(thetas, percentile_start), np.percentile(thetas, percentile_end)
    logger.debug(f'Q1 and Q3 for theta: {q1_theta}, {q3_theta}')
    q1_curvature, q3_curvature = np.percentile(curvatures, 10), np.percentile(curvatures, 90)
    logger.debug(f'Q1 and Q3 for curvature: {q1_curvature}, {q3_curvature}')

    # Keep all indices where theta is between these two elements (including them)
    selected_indices_based_on_theta = []
    middle_thetas = []
    for index in range(len(thetas)):
        theta = circular_motion_estimates[index].theta
        if (theta >= q1_theta) and (theta <= q3_theta):
            middle_thetas.append(theta)
            selected_indices_based_on_theta.append(index)
    logger.debug(f'Thetas within the specified range: {len(middle_thetas)} of {len(thetas)}')

    # Keep all indices where curvature is between these two elements (including them)
    selected_indices_based_on_curvature = []
    middle_curvatures = []
    for index in range(len(curvatures)):
        curvature = circular_motion_estimates[index].curvature
        if (curvature >= q1_curvature) and (curvature <= q3_curvature):
            middle_curvatures.append(curvature)
            selected_indices_based_on_curvature.append(index)
    logger.debug(f'Curvatures within the specified range: {len(middle_curvatures)} of {len(curvatures)}')

    # Find indices that are common between both middle ranges
    common_indices = list(set(selected_indices_based_on_theta).intersection(selected_indices_based_on_curvature))
    logger.debug(
        f'Indices common to both theta and curvature ranges: {len(common_indices)} of {len(circular_motion_estimates)}')

    cm_poses = []
    # for index in selected_indices_based_on_theta:
    # for index in selected_indices_based_on_curvature:
    for index in common_indices:
        radius = np.inf
        if circular_motion_estimates[index].curvature != 0:
            radius = 1 / circular_motion_estimates[index].curvature
        cm_poses.append(get_transform_by_r_and_theta(radius,
                                                     circular_motion_estimates[index].theta))

    dx_value = statistics.mean([motions[0, 3] for motions in cm_poses])
    dy_value = statistics.mean([motions[1, 3] for motions in cm_poses])
    dth_value = statistics.mean([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])

    return [dx_value, dy_value, dth_value]


def do_experimental_local_outlier_factor(circular_motion_estimates, idx):
    # Need to drop circular motion estimates where curvature values could be np.inf:
    validated_circular_motion_estimates = []
    for cme in circular_motion_estimates:
        if cme.curvature != np.inf:
            validated_circular_motion_estimates.append(cme)

    from sklearn.neighbors import LocalOutlierFactor
    cm_poses = []
    thetas = [cme.theta for cme in validated_circular_motion_estimates]
    curvatures = [cme.curvature for cme in validated_circular_motion_estimates]
    thetas = (thetas - np.mean(thetas)) / np.std(thetas)
    curvatures = (curvatures - np.ma.mean(curvatures)) / np.ma.std(curvatures)
    # if idx == 15:
    #     pdb.set_trace()
    X = np.transpose(np.ma.array([thetas, curvatures]))

    # fit the model for outlier detection (default)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.4)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    y_pred = clf.fit_predict(X)
    # pdb.set_trace()

    chosen_indices = np.where(y_pred == 1)[0]
    # bad_indices = np.where(y_pred == -1)

    # nice_points = circular_motion_estimates[cluster_indices]
    # pdb.set_trace()
    # logger.info(f'Using {len(chosen_indices)} out of {len(circular_motion_estimates)} circular motion estimates.')

    for idx in chosen_indices:
        radius = np.inf
        if validated_circular_motion_estimates[idx].curvature != 0:
            radius = 1 / validated_circular_motion_estimates[idx].curvature
        cm_poses.append(get_transform_by_r_and_theta(radius,
                                                     validated_circular_motion_estimates[idx].theta))
    # pdb.set_trace()
    dx_value = statistics.mean([motions[0, 3] for motions in cm_poses])
    dy_value = statistics.mean([motions[1, 3] for motions in cm_poses])
    dth_value = statistics.mean([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])

    return [dx_value, dy_value, dth_value]

    # # A staging area for some plotting
    # plt.figure(figsize=(10, 10))
    # theta_values = [circular_motion_estimates[i].theta for i in cluster_indices[0]]
    # curvature_values = [circular_motion_estimates[i].curvature for i in cluster_indices[0]]
    # bad_theta_values = [circular_motion_estimates[i].theta for i in bad_indices[0]]
    # bad_curvature_values = [circular_motion_estimates[i].curvature for i in bad_indices[0]]
    # plt.plot(theta_values, curvature_values, 'g.')
    # plt.plot(bad_theta_values, bad_curvature_values, 'r.')
    # plt.title("Clustering")
    # plt.grid()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.ylim(-1, 1)
    # # plt.xlim(-1, 1)
    # plt.savefig("%s%i%s" % (
    #     "/workspace/data/landmark-distortion/ro_state_pb_developing/"
    #     "circular_motion_dev/figs_circular_motion_estimation/clustering", idx, ".pdf"))
    # plt.close()


def do_experimental_cluster(circular_motion_estimates, idx):
    from sklearn.cluster import KMeans
    logger.info("Doing kmeans things...")

    thetas = [cme.theta for cme in circular_motion_estimates]
    curvatures = [cme.curvature for cme in circular_motion_estimates]
    thetas = (thetas - np.mean(thetas)) / np.std(thetas)
    curvatures = (curvatures - np.mean(curvatures)) / np.std(curvatures)

    X = np.transpose(np.array([thetas, curvatures]))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(kmeans.labels_)
    cluster_indices = np.where(kmeans.labels_ == 1)
    bad_indices = np.where(kmeans.labels_ == 0)

    # nice_points = circular_motion_estimates[cluster_indices]
    # pdb.set_trace()

    # A staging area for some plotting
    plt.figure(figsize=(10, 10))
    theta_values = [circular_motion_estimates[i].theta for i in cluster_indices[0]]
    curvature_values = [circular_motion_estimates[i].curvature for i in cluster_indices[0]]
    bad_theta_values = [circular_motion_estimates[i].theta for i in bad_indices[0]]
    bad_curvature_values = [circular_motion_estimates[i].curvature for i in bad_indices[0]]
    plt.plot(theta_values, curvature_values, 'g.')
    plt.plot(bad_theta_values, bad_curvature_values, 'r.')
    plt.title("Clustering")
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylim(-1, 1)
    # plt.xlim(-1, 1)
    plt.savefig("%s%i%s" % (
        "/workspace/data/landmark-distortion/ro_state_pb_developing/"
        "circular_motion_dev/figs_circular_motion_estimation/clustering", idx, ".pdf"))
    plt.close()


def get_experimental_dx_dy_dth_from_circular_motion_estimates(circular_motion_estimates):
    import scipy
    from scipy.stats import sem
    cm_poses = []
    chosen_indices = []
    thetas = [cme.theta for cme in circular_motion_estimates]
    # mean_theta = np.mean(thetas)
    sd_theta = np.std(thetas)
    # sem_theta = scipy.stats.sem(thetas)  # standard error of the mean
    mad_theta = scipy.stats.median_abs_deviation(thetas)  # median absolute deviation
    # pdb.set_trace()
    lower_theta_bound = np.mean(thetas) - sd_theta
    upper_theta_bound = np.mean(thetas) + sd_theta
    # lower_theta_bound = np.median(thetas) - mad_theta
    # upper_theta_bound = np.median(thetas) + mad_theta
    for i in range(len(circular_motion_estimates)):
        if (circular_motion_estimates[i].theta >= lower_theta_bound) and (
                circular_motion_estimates[i].theta <= upper_theta_bound):
            chosen_indices.append(i)
    logger.debug(f'Using {len(chosen_indices)} out of {len(circular_motion_estimates)} circular motion estimates.')

    # A staging area for some plotting
    # plt.figure(figsize=(10, 10))
    theta_values = [circular_motion_estimates[i].theta for i in chosen_indices]
    curvature_values = [circular_motion_estimates[i].curvature for i in chosen_indices]
    # plt.plot(theta_values, '.', label="theta")
    # plt.plot(curvature_values, '.', label="curvature")
    # plt.title("Thetas")
    # plt.grid()
    # plt.legend()
    # # plt.ylim(-1, 1)
    # # plt.xlim(-1, 1)
    # plt.savefig("%s" % (
    #     "/workspace/data/landmark-distortion/ro_state_pb_developing/"
    #     "circular_motion_dev/figs_circular_motion_estimation/outlier_detection.pdf"))
    # plt.close()

    for idx in chosen_indices:
        radius = np.inf
        if circular_motion_estimates[idx].curvature != 0:
            radius = 1 / circular_motion_estimates[idx].curvature
        cm_poses.append(get_transform_by_r_and_theta(radius,
                                                     circular_motion_estimates[idx].theta))
    # pdb.set_trace()
    dx_value = statistics.mean([motions[0, 3] for motions in cm_poses])
    dy_value = statistics.mean([motions[1, 3] for motions in cm_poses])
    dth_value = statistics.mean([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])

    return [dx_value, dy_value, dth_value]


def get_mean_dx_dy_dth_from_circular_motion_estimates_iqr(circular_motion_estimates):
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
    logger.debug(f'Using {len(cm_poses)} out of {len(circular_motion_estimates)} circular motion estimates.')

    dx_value = statistics.mean([motions[0, 3] for motions in cm_poses])
    dy_value = statistics.mean([motions[1, 3] for motions in cm_poses])
    dth_value = statistics.mean([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])

    return [dx_value, dy_value, dth_value]


def get_median_dx_dy_dth_from_circular_motion_estimates_iqr(circular_motion_estimates):
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

    # print("Indices of medians for dx, dy, dtheta:")
    # print(np.argsort([motions[0, 3] for motions in cm_poses])[len([motions[0, 3] for motions in cm_poses]) // 2])
    # print(np.argsort([motions[1, 3] for motions in cm_poses])[len([motions[1, 3] for motions in cm_poses]) // 2])
    # print(np.argsort([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses])[
    #           len([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses]) // 2])
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
    logger.info("Plotting pose estimate data...")

    figure_path = params.output_path + "figs_circular_motion_estimation/"
    output_path = Path(figure_path)
    # if output_path.exists() and output_path.is_dir():
    #     shutil.rmtree(output_path)
    # output_path.mkdir(parents=True)

    # Pose estimates from SVD on the full set of matches
    full_match_timestamps, full_match_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.output_path + "full_matches_poses.csv")
    svd_x = [float(item[0]) for item in full_match_x_y_th]
    svd_y = [float(item[1]) for item in full_match_x_y_th]
    svd_th = [float(item[2]) for item in full_match_x_y_th]

    # Pose estimates from inliers only
    cm_timestamps, cm_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.output_path + "cm_matches_poses.csv")
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


def plot_sorted_values(figure_path, index, circular_motion_estimates):
    thetas = [cme.theta for cme in circular_motion_estimates]
    curvatures = [cme.curvature for cme in circular_motion_estimates]

    # get Q1 and Q3 element from thetas and curvatures
    percentile_start, percentile_end = 30, 70
    q1_theta, q3_theta = np.percentile(thetas, percentile_start), np.percentile(thetas, percentile_end)
    logger.debug(f'Q1 and Q3 for theta: {q1_theta}, {q3_theta}')
    q1_curvature, q3_curvature = np.percentile(curvatures, percentile_start), np.percentile(curvatures, percentile_end)
    logger.debug(f'Q1 and Q3 for curvature: {q1_curvature}, {q3_curvature}')

    # Keep all indices where theta is between these two elements (including them)
    selected_indices_based_on_theta = []
    middle_thetas = []
    middle_curvatures_based_on_theta = []
    middle_thetas_based_on_curvature = []

    for i in range(len(thetas)):
        theta = circular_motion_estimates[i].theta
        if (theta >= q1_theta) and (theta <= q3_theta):
            middle_thetas.append(theta)
            middle_curvatures_based_on_theta.append(circular_motion_estimates[i].curvature)
            selected_indices_based_on_theta.append(i)
    logger.debug(f'Thetas within the specified range: {len(middle_thetas)} of {len(thetas)}')

    # Keep all indices where curvature is between these two elements (including them)
    selected_indices_based_on_curvature = []
    middle_curvatures = []
    for i in range(len(curvatures)):
        curvature = circular_motion_estimates[i].curvature
        if (curvature >= q1_curvature) and (curvature <= q3_curvature):
            middle_curvatures.append(curvature)
            middle_thetas_based_on_curvature.append(circular_motion_estimates[i].theta)
            selected_indices_based_on_curvature.append(i)
    logger.debug(f'Curvatures within the specified range: {len(middle_curvatures)} of {len(curvatures)}')

    # Find indices that are common between both middle ranges
    common_indices = list(set(selected_indices_based_on_theta).intersection(selected_indices_based_on_curvature))

    # plt.figure(figsize=(10, 10))
    # plt.plot(middle_thetas, 'b.', label="middle_thetas")
    # plt.plot(middle_curvatures_based_on_theta, 'r.', label="curvatures_based_on_thetas")
    # plt.plot(middle_curvatures, 'rx', label="middle_curvatures")
    # plt.plot(middle_thetas_based_on_curvature, 'bx', label="thetas_based_on_curvature")
    # plt.title("Theta vs curvature")
    # plt.grid()
    # plt.xlabel("Curvature")
    # plt.ylabel("Theta")
    # plt.ylim(-5, 5)
    # # plt.xlim(-1, 1)
    # plt.legend()
    # plt.savefig("%s%s%i%s" % (figure_path, "/curvature_theta_", index, ".pdf"))
    # plt.close()

    # Plotting final motion estimates #
    cm_poses_theta = []
    for idx in selected_indices_based_on_theta:
        radius = np.inf
        if circular_motion_estimates[idx].curvature != 0:
            radius = 1 / circular_motion_estimates[idx].curvature
        cm_poses_theta.append(get_transform_by_r_and_theta(radius,
                                                           circular_motion_estimates[idx].theta))

    cm_poses_curvature = []
    for idx in selected_indices_based_on_curvature:
        radius = np.inf
        if circular_motion_estimates[idx].curvature != 0:
            radius = 1 / circular_motion_estimates[idx].curvature
        cm_poses_curvature.append(get_transform_by_r_and_theta(radius,
                                                               circular_motion_estimates[idx].theta))

    cm_poses_both = []
    for idx in common_indices:
        radius = np.inf
        if circular_motion_estimates[idx].curvature != 0:
            radius = 1 / circular_motion_estimates[idx].curvature
        cm_poses_both.append(get_transform_by_r_and_theta(radius,
                                                          circular_motion_estimates[idx].theta))

    # Plot some Gaussians
    import scipy.stats as stats
    import math
    dx_values_theta = [motions[0, 3] for motions in cm_poses_theta]
    plt.figure(figsize=(10, 10))
    mu = np.mean(dx_values_theta)
    variance = np.var(dx_values_theta)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label="dx_theta")

    dx_values_curvature = [motions[0, 3] for motions in cm_poses_curvature]
    mu = np.mean(dx_values_curvature)
    variance = np.var(dx_values_curvature)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label="dx_curvature")

    dx_values_both = [motions[0, 3] for motions in cm_poses_both]
    mu = np.mean(dx_values_both)
    variance = np.var(dx_values_both)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label="dx_both")

    plt.grid()
    plt.legend()
    plt.savefig("%s%s%i%s" % (figure_path, "/gaussian_", index, ".pdf"))
    plt.close()

    # plt.figure(figsize=(10, 10))
    # plt.plot([motions[0, 3] for motions in cm_poses], 'b.', label="dx")
    # plt.plot([motions[1, 3] for motions in cm_poses], 'r.', label="dy")
    # plt.plot([np.arctan2(motions[1, 0], motions[0, 0]) for motions in cm_poses], 'g.', label="dtheta")
    # plt.title("Motions")
    # plt.grid()
    # # plt.xlabel("Curvature")
    # # plt.ylabel("Theta")
    # plt.ylim(-5, 5)
    # # plt.xlim(-1, 1)
    # plt.legend()
    # plt.savefig("%s%s%i%s" % (figure_path, "/estimates_", index, ".pdf"))
    # plt.close()


def plot_1d_kde_values(figure_path, index, circular_motion_estimates):
    # Need to drop circular motion estimates where curvature values could be np.inf:
    validated_circular_motion_estimates = []
    for cme in circular_motion_estimates:
        if cme.curvature != np.inf:
            validated_circular_motion_estimates.append(cme)
    bw = 0.001
    from sklearn.neighbors import KernelDensity
    thetas = np.array([cme.theta for cme in validated_circular_motion_estimates])[:, np.newaxis]
    curvatures = np.array([cme.curvature for cme in validated_circular_motion_estimates])[:, np.newaxis]
    kde_thetas = KernelDensity(kernel='gaussian', bandwidth=bw).fit(thetas)
    kde_curvatures = KernelDensity(kernel='gaussian', bandwidth=bw).fit(curvatures)

    x_dim = 0.5
    x = np.linspace(-x_dim, x_dim, 10000)[:, np.newaxis]
    theta_density = np.exp(kde_thetas.score_samples(x))
    curvature_density = np.exp(kde_curvatures.score_samples(x))

    # Get max for theta and curvature from KDEs:
    best_theta = x[int(np.argmax(np.exp(kde_thetas.score_samples(x))))]
    best_curvature = x[int(np.argmax(np.exp(kde_curvatures.score_samples(x))))]
    print("Best theta:", best_theta)
    print("Best curvature:", best_curvature)

    plt.figure(figsize=(10, 10))
    plt.plot(x, theta_density, label="thetas")
    plt.plot(x, curvature_density, label="curvatures")
    plt.title("KDE")
    plt.grid()
    plt.legend()
    plt.savefig("%s%s%i%s" % (figure_path, "/kde_", index, ".pdf"))
    plt.close()


def kde2D(x, y, bandwidth, dim, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    from sklearn.neighbors import KernelDensity
    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[-dim:dim:xbins, -dim:dim:ybins]
    # xx, yy = np.mgrid[x.min():x.max():xbins,
    #          y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def plot_2d_kde_values(figure_path, index, circular_motion_estimates):
    # Need to drop circular motion estimates where curvature values could be np.inf:
    validated_circular_motion_estimates = []
    for cme in circular_motion_estimates:
        if cme.curvature != np.inf:
            validated_circular_motion_estimates.append(cme)
    from sklearn.neighbors import KernelDensity
    thetas = np.array([cme.theta for cme in validated_circular_motion_estimates])
    curvatures = np.array([cme.curvature for cme in validated_circular_motion_estimates])

    # m1 = thetas
    # m2 = curvatures
    # x, y = m1 + m2, m1 - m2
    x, y = thetas, curvatures
    value_window_dimension = 0.5
    num_bins = 100
    cell_size = 2 * value_window_dimension / num_bins
    xx, yy, zz = kde2D(x, y, bandwidth=0.005, dim=value_window_dimension)

    # Find "hottest" cell in heatmap
    max_zz_indices = np.unravel_index(zz.argmax(), zz.shape)
    # Get theta and curvature thresholds from the cell's limits - this is a center-valued cell
    # Perhaps later it would be worth taking nearby cells too if necessary
    theta_min, theta_max = xx[max_zz_indices[0]][0] - (cell_size / 1), xx[max_zz_indices[0]][0] + (cell_size / 1)
    curvature_min, curvature_max = yy[0][max_zz_indices[1]] - cell_size / 1, yy[0][max_zz_indices[1]] + cell_size / 1
    print(max_zz_indices)
    print(theta_min, theta_max)
    print(curvature_min, curvature_max)
    print("Cell size:", cell_size)
    print(curvature_min - curvature_max)
    # pdb.set_trace()

    # cropped_x = [item for item in thetas if theta_min < item < theta_max]
    # cropped_y = [item for item in thetas if theta_min < item < theta_max]
    # cropped_y = [item for item in curvatures if curvature_min < item < curvature_max]
    raw_points = np.transpose(np.array([x, y]))
    # pdb.set_trace()
    cropped_points = np.array([item for item in raw_points if theta_min < item[0] < theta_max])
    cropped_points = np.array([item for item in cropped_points if curvature_min < item[1] < curvature_max])

    print("Raw/cropped points size:", raw_points.shape, cropped_points.shape)

    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, zz, shading='auto')
    plt.scatter(x, y, s=2, facecolor='white')
    plt.scatter(cropped_points[:, 0], cropped_points[:, 1], s=2, facecolor='r')
    plt.title("KDE")
    plt.grid()
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.savefig("%s%s%i%s" % (figure_path, "/kde_2d_", index, ".png"))
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
    figure_path = params.output_path + "figs_circular_motion_estimation/error_metrics/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    gt_se3s, gt_timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    gt_se3s = gt_se3s[settings.K_RADAR_INDEX_OFFSET:]

    # Pose estimates from full matches
    full_matches_timestamps, full_matches_x_y_th = get_timestamps_and_x_y_th_from_csv(
        params.output_path + "full_matches_poses.csv")
    full_matches_se3s = get_raw_se3s_from_x_y_th(full_matches_x_y_th)

    # Pose estimates from circular motion only
    cm_timestamps, cm_x_y_th = get_timestamps_and_x_y_th_from_csv(params.output_path + "cm_matches_poses.csv")
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
    output_path_for_metrics = Path(params.output_path + "visualised_metrics")
    if output_path_for_metrics.exists() and output_path_for_metrics.is_dir():
        shutil.rmtree(output_path_for_metrics)
    output_path_for_metrics.mkdir(parents=True)

    visualiser = TrajectoryVisualizer({"full_matches": tm_gt_fullmatches, "cm": tm_gt_cm})
    visualiser.plot_cum_norm_err(outfile="%s%s" % (output_path_for_metrics, "/cumulative_norm_errors.pdf"))
    # visualiser.plot_norm_err(outfile="/workspace/data/visualised_metrics_tmp/norm_errors.pdf")
    visualiser.plot_segment_errors(segs=segment_lengths,
                                   outfile="%s%s" % (output_path_for_metrics, "/segment_errors.pdf"))
    visualiser.plot_topdown(which_plane='yx',  # this is a custom flip to conform to MRG convention, instead of xy
                            outfile="%s%s" % (output_path_for_metrics, "/topdown.pdf"))


def print_trajectory_metrics(tm_gt_est, segment_lengths, data_name="this"):
    logger.info(f'\nTrajectory Metrics for {data_name} set:')
    # print("endpoint_error:", tm_gt_est.endpoint_error(segment_lengths))
    # print("segment_errors:", tm_gt_est.segment_errors(segment_lengths))
    # print("traj_errors:", tm_gt_est.traj_errors())
    # print("rel_errors:", tm_gt_est.rel_errors())
    # print("error_norms:", tm_gt_est.error_norms())
    logger.info(f'mean_err: {tm_gt_est.mean_err()}')
    # print("cum_err:", tm_gt_est.cum_err())
    logger.info(f'rms_err: {tm_gt_est.rms_err()}')


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--output_path', type=str, default="",
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

    # python circular_motion_estimator.py
    # --input_path "/workspace/data/landmark-distortion/ro_state_pb_developing/ro_state_files/"
    # --output_path "/workspace/data/landmark-distortion/ro_state_pb_developing/circular_motion_dev/"
    # --num_samples 2000

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    logger.info(f'Number of indices in this radar odometry state monolithic: {len(radar_state_mono)}')

    circular_motion_estimation(params, radar_state_mono)
    plot_csv_things(params)
    # get_metrics(params)


if __name__ == "__main__":
    main()

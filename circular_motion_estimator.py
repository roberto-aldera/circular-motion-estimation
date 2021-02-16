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
import settings
import pdb
# from pyslam.metrics import TrajectoryMetrics
from pose_tools.pose_utils import *
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb
from get_rigid_body_motion import get_motion_estimate_from_svd
from R_and_theta_utilities import get_relative_range_and_bearing_from_x_and_y, get_theta_and_radius_from_single_match
from kinematics import get_transform_by_r_and_theta

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


def circular_motion_estimation(params, radar_state_mono):
    figure_path = params.input_path + "figs_circular_motion_estimation/"
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
    poses_from_circular_motion = []
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
        plt.figure(figsize=(10, 10))

        # for match_idx in range(5):
        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            # if x1 == x2 and y1 == y2:
            #     print("\t\t\t*** x1 == x2 and y1 == y2 for idx:", match_idx)
            #     pdb.set_trace()
            plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
            matched_points.append([x1, x2, y1, y2])

        thetas = []
        radii = []
        all_motions = []

        x1s = []
        y1s = []
        x2s = []
        y2s = []
        for tmp_idx in range(len(matched_points)):
            x1 = matched_points[tmp_idx][3]
            y1 = matched_points[tmp_idx][1]
            x2 = matched_points[tmp_idx][2]
            y2 = matched_points[tmp_idx][0]

            # print(x1, y1, x2, y2)
            if x1 == x2 and y1 == y2:
                print("\t\t\t*** x1 == x2 and y1 == y2 for idx:", tmp_idx)
            else:
                r1, a1 = get_relative_range_and_bearing_from_x_and_y(relative_x=x1, relative_y=y1)
                r2, a2 = get_relative_range_and_bearing_from_x_and_y(relative_x=x2, relative_y=y2)
                theta, radius = get_theta_and_radius_from_single_match(d_1=r1, d_2=r2, phi_1=a1, phi_2=a2)

                circular_motion_pose = get_transform_by_r_and_theta(radius, theta)
                all_motions.append(circular_motion_pose)
                # print("x:", circular_motion_pose[0, 3])
                # print("y:", circular_motion_pose[1, 3])

                thetas.append(theta)
                radii.append(radius)
                x1s.append(x1)
                y1s.append(y1)
                x2s.append(x2)
                y2s.append(y2)
                # pdb.set_trace()

        # # Save points for debugging.
        # with open("/workspace/data/landmark-distortion/ro_state_pb_developing/points.csv", 'w') as points_file:
        #     wr = csv.writer(points_file, delimiter=",")
        #     for idx in range(len(x1s)):
        #         wr.writerow([x1s[idx], y1s[idx], x2s[idx], y2s[idx]])

        # find medians
        theta_median = statistics.median(thetas)
        # radius_median = statistics.median(radii)
        print("theta median:", theta_median)
        # print("radius median:", radius_median)

        #############################################################################
        theta_median_idx = np.argsort(thetas)[len(thetas) // 2]
        print("theta median index:", theta_median_idx)
        print("theta median:", thetas[theta_median_idx])
        # Now get the theta and radius corresponding to this median value for theta:
        x1 = matched_points[theta_median_idx][3]
        y1 = matched_points[theta_median_idx][1]
        x2 = matched_points[theta_median_idx][2]
        y2 = matched_points[theta_median_idx][0]

        # print(x1, y1, x2, y2)

        r1, a1 = get_relative_range_and_bearing_from_x_and_y(relative_x=x1, relative_y=y1)
        r2, a2 = get_relative_range_and_bearing_from_x_and_y(relative_x=x2, relative_y=y2)
        theta, radius = get_theta_and_radius_from_single_match(d_1=r1, d_2=r2, phi_1=a1, phi_2=a2)

        circular_motion_pose = get_transform_by_r_and_theta(radius, theta)
        poses_from_circular_motion.append([circular_motion_pose[0, 3], circular_motion_pose[1, 3],
                                           np.arctan2(circular_motion_pose[1, 0], circular_motion_pose[0, 0])])

        # pdb.set_trace()
        #############################################################################

        # find transforms
        # circular_motion_pose = get_transform_by_r_and_theta(rotation_radius=radius_median, theta=theta_median)
        print("x:", circular_motion_pose[0, 3])
        print("y:", circular_motion_pose[1, 3])

        # pdb.set_trace()
        # plt.figure(figsize=(10, 10))
        # n, bins, patches = plt.hist(radii, 1000, density=False, facecolor='tab:blue')
        plt.plot(y1s, x1s, "*")
        plt.plot(y2s, x2s, "*")
        plt.title("Title")
        plt.grid()
        # plt.gca().set_aspect('equal', adjustable='box')
        # dim = 5
        # plt.xlim(-dim, dim)
        # plt.ylim(-dim, dim)
        plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/debugging.pdf")
        plt.close()

        # plt.figure(figsize=(10, 10))
        # # plt.plot(np.abs(radii), '.')
        # n, bins, patches = plt.hist(radii, 100, density=False, facecolor='tab:blue')
        # plt.title("Radius values")
        # plt.grid()
        # # plt.gca().set_aspect('equal', adjustable='box')
        # # dim = 5
        # # plt.xlim(0, 5000)
        # # plt.ylim(-2000,2000)
        # plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/debugging_radii.pdf")
        # plt.close()

        plt.figure(figsize=(10, 10))
        # plt.plot(np.abs(thetas), '.')
        n, bins, patches = plt.hist(thetas, 100, density=False, facecolor='tab:blue')
        plt.title("Theta values")
        plt.grid()
        # plt.gca().set_aspect('equal', adjustable='box')
        # dim = 5
        # plt.xlim(0, 5000)
        # plt.ylim(-2000,2000)
        plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/debugging_thetas.pdf")
        plt.close()

        # plt.figure(figsize=(10, 10))
        # dx_vals = [motions[0, 3] for motions in all_motions]
        # print("dx median:", statistics.median(dx_vals))
        # plt.hist(dx_vals, 100, density=False, facecolor='tab:blue')
        # plt.title("dx values")
        # plt.grid()
        # plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/debugging_dx_vals.pdf")
        # plt.close()

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
        pose_from_svd = [v[1], v[0], -theta_R]  # this line applies the transform to get into the robot frame
        poses_from_full_match_set.append(pose_from_svd)
        print("SVD motion estimate (x, y, th):", pose_from_svd)

    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_full_match_set,
                                      pose_source="full_matches",
                                      export_folder=params.input_path)
    save_timestamps_and_x_y_th_to_csv(timestamps_from_ro_state, x_y_th=poses_from_circular_motion,
                                      pose_source="cm_matches",
                                      export_folder=params.input_path)


def plot_csv_things(params):
    print("Plotting pose estimate data...")

    figure_path = params.input_path + "figs_circular_motion_estimation/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

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
    plot_csv_things(params)


if __name__ == "__main__":
    main()

    # python - W error circular_motion_estimator.py
    # --input_path "/workspace/data/landmark-distortion/ro_state_pb_developing/" - -num_samples 4
    # try:
    #     main()
    # except:
    #     type, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
    #     frame = last_frame().tb_frame
    #     ns = dict(frame.f_globals)
    #     ns.update(frame.f_locals)
    #     code.interact(local=ns)

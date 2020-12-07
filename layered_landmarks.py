import numpy as np
import matplotlib.pyplot as plt
import sys
from pose_tools.pose_utils import *
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings


def render_landmarks(params):
    landmarks_path = params.landmarks_path
    figure_path = landmarks_path + "figs_layered_landmarks/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # se3s, timestamps = get_poses_from_file(params.relative_poses)
    se3s, timestamps = get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")
    # print(se3s[0])
    global_pose = np.eye(4)

    plt.figure(figsize=(20, 20))
    # dim = 200
    # plt.xlim(-dim, dim)
    # plt.ylim(-dim, dim)

    k_inner_range = 20
    k_every_nth_scan = 25
    plot_as_inner_and_outer_landmarks = False

    for i in range(params.num_samples):
        folder_path = landmarks_path + "landmarks_" + str(i)
        landmarks_data_from_csv_with_timestamp = np.genfromtxt(folder_path + ".csv", delimiter=",")
        landmarks_timestamp = landmarks_data_from_csv_with_timestamp[0, 0]
        raw_landmarks = landmarks_data_from_csv_with_timestamp[1:, :]
        landmarks = np.c_[
            raw_landmarks, np.zeros(len(raw_landmarks)), np.ones(
                len(raw_landmarks))]  # so that se3 multiplication works
        if i > 0:
            relative_pose_index = i - 1 + 141
            relative_pose_timestamp = timestamps[relative_pose_index]

            # ensure timestamps are within a reasonable limit of each other (microseconds)
            assert (landmarks_timestamp - relative_pose_timestamp) < 500

            global_pose = global_pose @ se3s[relative_pose_index]
            landmarks = np.transpose(global_pose @ np.transpose(landmarks))
            if i % k_every_nth_scan == 0:
                print("Processing index: ", i)

                # plot x and y swapped around so that robot is moving forward as upward direction
                if plot_as_inner_and_outer_landmarks:
                    for j in range(len(landmarks)):
                        this_landmark_range = np.sqrt(pow(raw_landmarks[j, 1], 2) + pow(raw_landmarks[j, 0], 2))
                        if this_landmark_range < k_inner_range:
                            plt.plot(landmarks[j, 1], landmarks[j, 0], 'r,')  # use ',' for pixel markers
                        else:
                            plt.plot(landmarks[j, 1], landmarks[j, 0], 'k,')  # use ',' for pixel markers
                    plt.plot(global_pose[1, 3], global_pose[0, 3], 'b^')
                else:
                    p = plt.plot(landmarks[:, 1], landmarks[:, 0], '+', markerfacecolor='none', markersize=1)
                    # plt.plot(landmarks[:, 1], landmarks[:, 0], ',')  # use ',' for pixel markers
                    plt.plot(global_pose[1, 3], global_pose[0, 3], '^', color=p[-1].get_color())

    # plot sensor range for Oxford radar robotcar dataset
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    r = 163
    x1 = global_pose[1, 3] + r * np.cos(circle_theta)
    x2 = global_pose[0, 3] + r * np.sin(circle_theta)
    plt.plot(x1, x2, 'g--')

    # x1 = global_pose[1, 3] + k_inner_range * np.cos(circle_theta)
    # x2 = global_pose[0, 3] + k_inner_range * np.sin(circle_theta)
    # plt.plot(x1, x2, 'r--')

    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("%s%s%s" % (output_path, "/landmarks_layered", ".png"))
    plt.close()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--relative_poses', type=str, default="", help='Path to relative pose file')
    parser.add_argument('--landmarks_path', type=str, default="",
                        help='Path to landmarks that were exported for processing')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES, help='Number of samples to process')
    params = parser.parse_args()

    print("Running landmark layering...")

    # get a landmark set in and plot it
    render_landmarks(params)


if __name__ == "__main__":
    main()

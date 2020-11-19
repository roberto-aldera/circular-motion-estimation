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
    figure_path = landmarks_path + "figs/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # th = 0
    # th = np.pi / 4 + np.pi  # roughly the angle the radar is offset by
    # rotation_matrix = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    for i in range(params.num_samples):
        folder_path = landmarks_path + "landmarks_" + str(i)
        print(folder_path)
        landmarks = np.genfromtxt(folder_path + ".csv", delimiter=",")
        # print(landmarks)
        # landmarks = landmarks @ rotation_matrix
        # landmarks[:, 0] = -landmarks[:, 0]
        landmarks[:, 0], landmarks[:, 1] = np.array(landmarks[:, 1]), np.array(landmarks[:, 0])
        # landmarks[:, 0], landmarks[:, 1] = b, a

        plt.figure(figsize=(20, 20))
        dim = 200
        plt.xlim(-dim, dim)
        plt.ylim(-dim, dim)
        plt.plot(landmarks[:, 0], landmarks[:, 1], "*")
        plt.grid()
        plt.savefig("%s%s%i%s" % (output_path, "/landmarks_", i, ".png"))
        plt.close()


def plot_velocity_estimates():
    poses_path = "/workspace/data/radar-tmp/ro_relative_poses.monolithic"
    se3s, timestamps = get_poses_from_file(poses_path)
    x_velocities, y_velocities, th_velocities = get_x_y_th_velocities_from_poses(se3s, timestamps)
    plt.figure(figsize=(10, 10))
    plt.plot(x_velocities, '.-', label="x")
    plt.plot(y_velocities, '.-', label="y")
    plt.plot(th_velocities, '.-', label="th")
    plt.grid()
    plt.legend()
    plt.title("RO velocity estimates")
    plt.xlabel("Sample index")
    plt.ylabel("Velocity (m/s)")
    plt.savefig("speeds.png")
    plt.close()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--landmarks_path', type=str, default="",
                        help='Path to landmarks that were exported for processing')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES, help='Number of samples to process')
    params = parser.parse_args()

    print("Running landmark renderer...")
    render_landmarks(params)


if __name__ == "__main__":
    main()

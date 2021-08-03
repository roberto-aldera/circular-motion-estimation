import numpy as np
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
import pdb
from pyslam.metrics import TrajectoryMetrics
import pandas as pd
import csv
from liegroups import SE3


def make_plot(params, gt_x_y_th, aux0_x_y_th, aux1_x_y_th):
    figure_path = Path(params.path) / "figs_odometry"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    x0 = []
    y0 = []
    th0 = []
    for sample in aux0_x_y_th:
        x0.append(float(sample[0]))
        y0.append(float(sample[1]))
        th0.append(float(sample[2]))

    x1 = []
    y1 = []
    th1 = []
    for sample in aux1_x_y_th:
        x1.append(float(sample[0]))
        y1.append(float(sample[1]))
        th1.append(float(sample[2]))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    start_idx = 0
    plt.xlim(start_idx, start_idx + params.num_samples)
    plt.grid()
    m_size = 3
    line_width = 0.3
    mew = 0.5  # marker edge width
    # plt.plot(np.array(gt_x_y_th[0]), '+-', color="black", linewidth=line_width, markersize=m_size, mew=mew,
    #          label="dx_GT")
    plt.plot(np.array(gt_x_y_th[1]), 'x-', color="black", linewidth=line_width, markersize=m_size, mew=mew,
             label="dy_GT")
    plt.plot(np.array(gt_x_y_th[2]), '.-', color="black", linewidth=line_width, markersize=m_size, mew=mew,
             fillstyle="none",
             label="dth_GT")
    # plt.plot(np.array(x0), '+-', color="tab:red", linewidth=line_width, markersize=m_size, mew=mew, label="dx_RO")
    plt.plot(np.array(y0), 'x-', color="tab:red", linewidth=line_width, markersize=m_size, mew=mew, label="dy_RO")
    plt.plot(np.array(th0), '.-', color="tab:red", linewidth=line_width, markersize=m_size, mew=mew, fillstyle="none",
             label="dth_RO")
    # plt.plot(np.array(x1), '+-', color="tab:blue", linewidth=line_width, markersize=m_size, mew=mew, label="dx_CC")
    plt.plot(np.array(y1), 'x-', color="tab:blue", linewidth=line_width, markersize=m_size, mew=mew, label="dy_CC")
    plt.plot(np.array(th1), '.-', color="tab:blue", linewidth=line_width, markersize=m_size, mew=mew, fillstyle="none",
             label="dth_CC")
    plt.title("Pose estimate performance for lateral and yaw motion")
    plt.xlabel("Sample index")
    plt.ylabel("m/sample, rad/sample")
    plt.legend()
    plt.tight_layout()
    figure_path = "%s%s" % (output_path, "/xyth_comparison.pdf")
    plt.savefig(figure_path)
    plt.close()
    print("Saved figure to:", figure_path)


def get_ground_truth_poses_from_csv(path_to_gt_csv):
    """
    Load poses from csv for the Oxford radar robotcar 10k dataset.
    """
    df = pd.read_csv(path_to_gt_csv)
    # print(df.head())
    x_vals = df['x']
    y_vals = df['y']
    th_vals = df['yaw']
    timestamps = df['source_radar_timestamp']
    x_y_th = [x_vals, y_vals, th_vals]

    se3s = []
    for i in range(len(df.index)):
        th = th_vals[i]
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = x_vals[i]
        pose[1, 3] = y_vals[i]
        se3s.append(pose)
    return se3s, timestamps, x_y_th


def get_timestamps_and_x_y_th_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        pose_data = list(reader)

    timestamps = [int(item[0]) for item in pose_data]
    x_y_th = [items[1:] for items in pose_data]
    return timestamps, x_y_th


def get_timestamps_and_x_y_th_from_circular_motion_estimate_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        pose_data = list(reader)

    # timestamps = [int(item[0]) for item in pose_data]
    timestamps = [int(0) for item in pose_data]
    x_y_th = [items[3:] for items in pose_data]
    return timestamps, x_y_th


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--path', type=str, default="",
                        help='Path to folder where inputs are and where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running script...")

    gt_se3s, gt_timestamps, gt_x_y_th = get_ground_truth_poses_from_csv(
        "/workspace/data/ro-state-files/radar_oxford_10k/2019-01-10-14-50-05/radar_odometry.csv")
    # gt_se3s = gt_se3s[settings.K_RADAR_INDEX_OFFSET:]

    _, aux0_x_y_th = get_timestamps_and_x_y_th_from_csv(
        "/workspace/data/landmark-distortion/final-results/2019-01-10-14-50-05/full_matches_poses.csv")
    _, aux1_x_y_th = get_timestamps_and_x_y_th_from_csv(
        "/workspace/data/landmark-distortion/final-results/2019-01-10-14-50-05/cm_matches_poses.csv")

    make_plot(params, gt_x_y_th, aux0_x_y_th, aux1_x_y_th)


if __name__ == "__main__":
    main()

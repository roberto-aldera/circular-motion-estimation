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


def get_metrics(params):
    folder_date_name = "2019-01-10-15-19-41"
    # Some code to run KITTI metrics over poses, based on pyslam TrajectoryMetrics
    gt_se3s, gt_timestamps = get_ground_truth_poses_from_csv(
        "/Users/roberto/data/ro-state-files/radar_oxford_10k/" + folder_date_name + "/radar_odometry.csv")
    gt_se3s = gt_se3s[settings.K_RADAR_INDEX_OFFSET:]

    num_iterations = min(params.num_samples, len(gt_se3s))

    # Pose estimates from full matches
    full_matches_timestamps, full_matches_x_y_th = get_timestamps_and_x_y_th_from_csv(
        "/Users/roberto/data/landmark-distortion/final-results/" + folder_date_name + "/full_matches_poses.csv")

    # CME metrics vs RANSAC
    _, aux0_x_y_th = get_timestamps_and_x_y_th_from_csv(
        "/Users/roberto/data/landmark-distortion/final-results/" + folder_date_name + "/ransac_poses.csv")
    _, aux1_x_y_th = get_timestamps_and_x_y_th_from_csv(
        "/Users/roberto/data/landmark-distortion/final-results/" + folder_date_name + "/35-65-percentiles/cm_matches_svd_poses.csv")
    _, aux2_x_y_th = get_timestamps_and_x_y_th_from_csv(
        "/Users/roberto/data/landmark-distortion/final-results/" + folder_date_name + "/35-65-percentiles/cm_matches_poses.csv")

    # Cropping if necessary
    full_matches_timestamps, full_matches_x_y_th = full_matches_timestamps[:num_iterations], \
                                                   full_matches_x_y_th[:num_iterations]
    aux0_x_y_th = aux0_x_y_th[:num_iterations]
    aux1_x_y_th = aux1_x_y_th[:num_iterations]
    aux2_x_y_th = aux2_x_y_th[:num_iterations]

    # Just a sanity check here
    # do_quick_debugging_plot(aux0_x_y_th, aux1_x_y_th, params)

    full_matches_se3s = get_raw_se3s_from_x_y_th(full_matches_x_y_th)
    aux0_se3s = get_raw_se3s_from_x_y_th(aux0_x_y_th)
    aux1_se3s = get_raw_se3s_from_x_y_th(aux1_x_y_th)
    aux2_se3s = get_raw_se3s_from_x_y_th(aux2_x_y_th)

    relative_pose_index = settings.K_RADAR_INDEX_OFFSET + 1
    relative_pose_timestamp = gt_timestamps[relative_pose_index]

    # ensure timestamps are within a reasonable limit of each other (microseconds)
    assert (full_matches_timestamps[0] - relative_pose_timestamp) < 500
    # assert (aux0_timestamps[0] - relative_pose_timestamp) < 500
    # assert (aux1_timestamps[0] - relative_pose_timestamp) < 500

    # making global poses from the relative poses
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_global_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    fm_global_se3s = [np.identity(4)]
    for i in range(1, len(full_matches_se3s)):
        fm_global_se3s.append(fm_global_se3s[i - 1] @ full_matches_se3s[i])
    full_matches_global_SE3s = get_se3s_from_raw_se3s(fm_global_se3s)

    aux0_global_se3s = [np.identity(4)]
    for i in range(1, len(aux0_se3s)):
        aux0_global_se3s.append(aux0_global_se3s[i - 1] @ aux0_se3s[i])
    aux0_global_SE3s = get_se3s_from_raw_se3s(aux0_global_se3s)

    aux1_global_se3s = [np.identity(4)]
    for i in range(1, len(aux1_se3s)):
        aux1_global_se3s.append(aux1_global_se3s[i - 1] @ aux1_se3s[i])
    aux1_global_SE3s = get_se3s_from_raw_se3s(aux1_global_se3s)

    aux2_global_se3s = [np.identity(4)]
    for i in range(1, len(aux2_se3s)):
        aux2_global_se3s.append(aux2_global_se3s[i - 1] @ aux2_se3s[i])
    aux2_global_SE3s = get_se3s_from_raw_se3s(aux2_global_se3s)

    segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    tm_gt_fullmatches = TrajectoryMetrics(gt_global_SE3s, full_matches_global_SE3s)
    # print_trajectory_metrics(tm_gt_fullmatches, segment_lengths, data_name="full match")

    tm_gt_aux0 = TrajectoryMetrics(gt_global_SE3s, aux0_global_SE3s)
    # print_trajectory_metrics(tm_gt_aux0, segment_lengths, data_name=settings.AUX0_NAME)

    tm_gt_aux1 = TrajectoryMetrics(gt_global_SE3s, aux1_global_SE3s)
    # print_trajectory_metrics(tm_gt_aux1, segment_lengths, data_name=settings.AUX1_NAME)

    tm_gt_aux2 = TrajectoryMetrics(gt_global_SE3s, aux2_global_SE3s)
    # print_trajectory_metrics(tm_gt_aux2, segment_lengths, data_name=settings.AUX2_NAME)

    # save_trajectory_metrics_to_file(params, {"Full RO": tm_gt_fullmatches, settings.AUX0_NAME: tm_gt_aux0,
    #                                          settings.AUX1_NAME: tm_gt_aux1, settings.AUX2_NAME: tm_gt_aux2},
    #                                 segment_lengths)


    # Visualiser experimenting
    from pyslam.visualizers import TrajectoryVisualizer
    output_path_for_metrics = Path(params.path + "visualised_metrics")
    if output_path_for_metrics.exists() and output_path_for_metrics.is_dir():
        shutil.rmtree(output_path_for_metrics)
    output_path_for_metrics.mkdir(parents=True)

    visualiser = TrajectoryVisualizer(
        {"Full matches": tm_gt_fullmatches, settings.AUX0_NAME: tm_gt_aux0, settings.AUX1_NAME: tm_gt_aux1,
         settings.AUX2_NAME: tm_gt_aux2})
    visualiser.plot_cum_norm_err(figsize=(10, 3),
                                 outfile="%s%s" % (output_path_for_metrics, "/cumulative_norm_errors.pdf"))
    visualiser.plot_segment_errors(figsize=(10, 4), segs=segment_lengths, legend_fontsize=8,
                                   outfile="%s%s" % (output_path_for_metrics, "/segment_errors.pdf"))
    visualiser.plot_topdown(which_plane='yx',  # this is a custom flip to conform to MRG convention, instead of xy
                            outfile="%s%s" % (output_path_for_metrics, "/topdown.pdf"), figsize=(6, 6))
    # ,legend_fontsize=8, fontsize=16)


def save_trajectory_metrics_to_file(params, tm_gt_est_dict, segment_lengths):
    print("Calculating trajectory metrics to save to file...")
    results_file = Path(params.path + "results.txt")
    with open(results_file, "w") as text_file:
        for data_name, tm_gt_est in tm_gt_est_dict.items():
            print(f"{data_name} matches:", file=text_file)
            print(
                f"Segment error - lengths (m), translation (m), rotation (deg) \n {tm_gt_est.segment_errors(segment_lengths, rot_unit='deg')[1]}",
                file=text_file)
            print(
                f"Mean segment error: translation (m), rotation (deg) \n {np.mean(tm_gt_est.segment_errors(segment_lengths, rot_unit='deg')[1], axis=0)[1:]} \n",
                file=text_file)


def print_trajectory_metrics(tm_gt_est, segment_lengths, data_name="this"):
    print("\nTrajectory Metrics for", data_name, "set:")
    # print("endpoint_error:", tm_gt_est.endpoint_error(segment_lengths))
    # print("segment_errors:", tm_gt_est.segment_errors(segment_lengths)[1])
    print("average segment_error:", np.mean(tm_gt_est.segment_errors(segment_lengths, rot_unit='deg')[1], axis=0)[1:])
    # print("traj_errors:", tm_gt_est.traj_errors())
    # print("rel_errors:", tm_gt_est.rel_errors())
    # print("error_norms:", tm_gt_est.error_norms())
    print("mean_err:", tm_gt_est.mean_err(rot_unit='deg'))
    # print("traj cum_err:", tm_gt_est.cum_err(error_type='traj'))
    # print("rel cum_err:", tm_gt_est.cum_err(error_type='rel'))
    print("rms_err:", tm_gt_est.rms_err(rot_unit='deg'))


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
    return se3s, timestamps


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


def get_se3s_from_raw_se3s(raw_se3s):
    """
    Transform from raw se3 matrices into fancier SE3 type
    """
    se3s = []
    for pose in raw_se3s:
        se3s.append(SE3.from_matrix(np.asarray(pose)))
    return se3s


def get_raw_se3s_from_x_y_th(x_y_th):
    """
    Transform from x_y_th list into raw SE3 type
    """
    se3s = []
    for sample in x_y_th:
        th = float(sample[2])
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = float(sample[0])
        pose[1, 3] = float(sample[1])
        se3s.append(pose)
    return se3s


def do_quick_debugging_plot(aux0_x_y_th, aux1_x_y_th, params):
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
    plt.figure(figsize=(15, 5))
    dim = settings.TOTAL_SAMPLES + 50
    plt.xlim(0, 1000)
    plt.ylim(-1, 5)
    plt.grid()
    m_size = 5
    line_width = 0.5
    # plt.plot(np.array(gt_x_y_th[0]), 'b+-', linewidth=line_width, markersize=m_size, mew=0.3, label="dx_gt")
    # plt.plot(np.array(gt_x_y_th[1]), 'bx-', linewidth=line_width, markersize=m_size, mew=0.3, label="dy_gt")
    # plt.plot(np.array(gt_x_y_th[2]), 'bo-', linewidth=line_width, markersize=m_size, mew=0.3, fillstyle="none",
    #          label="dth_gt")
    plt.plot(np.array(x0), 'r+-', linewidth=line_width, markersize=m_size, mew=0.3, label="dx_raw")
    plt.plot(np.array(y0), 'rx-', linewidth=line_width, markersize=m_size, mew=0.3, label="dy_raw")
    plt.plot(np.array(th0), 'ro-', linewidth=line_width, markersize=m_size, mew=0.3, fillstyle="none",
             label="dth_raw")
    plt.plot(np.array(x1), 'g+-', linewidth=line_width, markersize=m_size, mew=0.3, label="dx_pred")
    plt.plot(np.array(y1), 'gx-', linewidth=line_width, markersize=m_size, mew=0.3, label="dy_pred")
    plt.plot(np.array(th1), 'go-', linewidth=line_width, markersize=m_size, mew=0.3, fillstyle="none",
             label="dth_pred")
    plt.title("Pose estimates")
    plt.xlabel("Sample index")
    plt.ylabel("units/sample")
    plt.legend()
    quick_figure_for_debugging_path = "%s%s" % (params.path, "quick_check_pose_predictions_comparison.pdf")
    plt.savefig(quick_figure_for_debugging_path)
    plt.close()
    print("Saved figure to:", quick_figure_for_debugging_path)


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--path', type=str, default="",
                        help='Path to folder where inputs are and where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running script...")
    # python generate_metrics.py --path "/workspace/data/landmark-dewarping/metrics/" --num_samples 2000

    get_metrics(params)


if __name__ == "__main__":
    main()

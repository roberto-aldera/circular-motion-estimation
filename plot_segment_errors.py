import numpy as np
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
import pdb
import csv


def make_plot(params, ro_trans_err, ro_rot_err, ransac_trans_err, ransac_rot_err, cc_svd_trans_err, cc_svd_rot_err,
              cc_means_trans_err, cc_means_rot_err):
    figure_path = Path(params.path) / "figs_segment_errors"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    alpha = 0.25

    ro_trans_means = np.mean(ro_trans_err, axis=0)
    ro_trans_std = np.std(ro_trans_err, axis=0)
    ro_rot_means = np.mean(ro_rot_err, axis=0)
    ro_rot_std = np.std(ro_rot_err, axis=0)
    print("RO mean translational error:", np.mean(ro_trans_means))
    print("RO mean rotational error:", np.mean(ro_rot_means))

    ransac_trans_means = np.mean(ransac_trans_err, axis=0)
    ransac_trans_std = np.std(ransac_trans_err, axis=0)
    ransac_rot_means = np.mean(ransac_rot_err, axis=0)
    ransac_rot_std = np.std(ransac_rot_err, axis=0)
    print("RANSAC mean translational error:", np.mean(ransac_trans_means))
    print("RANSAC mean rotational error:", np.mean(ransac_rot_means))

    cc_svd_trans_means = np.mean(cc_svd_trans_err, axis=0)
    cc_svd_trans_std = np.std(cc_svd_trans_err, axis=0)
    cc_svd_rot_means = np.mean(cc_svd_rot_err, axis=0)
    cc_svd_rot_std = np.std(cc_svd_rot_err, axis=0)
    print("CC-SVD mean translational error:", np.mean(cc_svd_trans_means))
    print("CC-SVD mean rotational error:", np.mean(cc_svd_rot_means))

    cc_mean_trans_means = np.mean(cc_means_trans_err, axis=0)
    cc_mean_trans_std = np.std(cc_means_trans_err, axis=0)
    cc_mean_rot_means = np.mean(cc_means_rot_err, axis=0)
    cc_mean_rot_std = np.std(cc_means_rot_err, axis=0)
    print("CC-means mean translational error:", np.mean(cc_mean_trans_means))
    print("CC-means mean rotational error:", np.mean(cc_mean_rot_means))

    import matplotlib.pyplot as plt
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(segment_lengths, ro_trans_means, "^-", color="tab:blue", label="RO")
    ax[0].fill_between(segment_lengths, ro_trans_means - ro_trans_std, ro_trans_means + ro_trans_std, color="tab:blue",
                       alpha=alpha)
    ax[0].plot(segment_lengths, ransac_trans_means, "^-", color="tab:orange", label="RANSAC")
    ax[0].fill_between(segment_lengths, ransac_trans_means - ransac_trans_std, ransac_trans_means + ransac_trans_std,
                       color="tab:orange", alpha=alpha)
    ax[0].plot(segment_lengths, cc_svd_trans_means, "^-", color="tab:green", label="CC-SVD")
    ax[0].fill_between(segment_lengths, cc_svd_trans_means - cc_svd_trans_std, cc_svd_trans_means + cc_svd_trans_std,
                       color="tab:green", alpha=alpha)
    ax[0].plot(segment_lengths, cc_mean_trans_means, "^-", color="tab:red", label="CC-means")
    ax[0].fill_between(segment_lengths, cc_mean_trans_means - cc_mean_trans_std,
                       cc_mean_trans_means + cc_mean_trans_std, color="tab:red", alpha=alpha)

    ax[0].grid()
    ax[0].set_title("Translational error")
    ax[0].set_xlabel("Segment length (m)")
    ax[0].set_ylabel("Average error (%)")
    ax[0].legend()

    ax[1].plot(segment_lengths, ro_rot_means, "^-", color="tab:blue", label="RO")
    ax[1].fill_between(segment_lengths, ro_rot_means - ro_rot_std, ro_rot_means + ro_rot_std, color="tab:blue",
                       alpha=alpha)
    ax[1].plot(segment_lengths, ransac_rot_means, "^-", color="tab:orange", label="RANSAC")
    ax[1].fill_between(segment_lengths, ransac_rot_means - ransac_rot_std, ransac_rot_means + ransac_rot_std,
                       color="tab:orange", alpha=alpha)
    ax[1].plot(segment_lengths, cc_svd_rot_means, "^-", color="tab:green", label="CC-SVD")
    ax[1].fill_between(segment_lengths, cc_svd_rot_means - cc_svd_rot_std, cc_svd_rot_means + cc_svd_rot_std,
                       color="tab:green", alpha=alpha)
    ax[1].plot(segment_lengths, cc_mean_rot_means, "^-", color="tab:red", label="CC-means")
    ax[1].fill_between(segment_lengths, cc_mean_rot_means - cc_mean_rot_std, cc_mean_rot_means + cc_mean_rot_std,
                       color="tab:red", alpha=alpha)

    ax[1].grid()
    ax[1].set_title("Rotational error")
    ax[1].set_xlabel("Segment length (m)")
    ax[1].set_ylabel("Average error (deg/m)")

    fig.tight_layout()
    figure_path = "%s%s" % (output_path, "/segment_errors.pdf")
    fig.savefig(figure_path)
    plt.close()
    print("Saved figure to:", figure_path)


def get_segment_errors_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = data[1:]  # discard header

    trans_err = []
    rot_err = []

    for item in data[:5]:  # first 5 rows are translational
        trans_err.append([float(i) for i in item[1:]])
    for item in data[5:]:  # 5-10 rows are rotational
        rot_err.append([float(i) for i in item[1:]])

    # *100 to convert to percentages for translation
    return np.array(trans_err) * 100, np.array(rot_err)


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--path', type=str, default="",
                        help='Path to folder where outputs will be saved')
    params = parser.parse_args()

    print("Running script...")

    segment_errors_folder = "/workspace/data/landmark-distortion/final-results/segment-errors/"
    ro_errors_file = segment_errors_folder + "ro.csv"
    ransac_errors_file = segment_errors_folder + "ransac.csv"
    cc_svd_errors_file = segment_errors_folder + "35-65-percentiles/cc-svd.csv"
    cc_means_errors_file = segment_errors_folder + "35-65-percentiles/cc-means.csv"

    ro_trans_err, ro_rot_err = get_segment_errors_from_csv(ro_errors_file)
    ransac_trans_err, ransac_rot_err = get_segment_errors_from_csv(ransac_errors_file)
    cc_svd_trans_err, cc_svd_rot_err = get_segment_errors_from_csv(cc_svd_errors_file)
    cc_means_trans_err, cc_means_rot_err = get_segment_errors_from_csv(cc_means_errors_file)
    make_plot(params, ro_trans_err, ro_rot_err, ransac_trans_err, ransac_rot_err, cc_svd_trans_err, cc_svd_rot_err,
              cc_means_trans_err, cc_means_rot_err)


if __name__ == "__main__":
    main()

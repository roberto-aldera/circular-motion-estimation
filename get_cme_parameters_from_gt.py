from pose_tools.pose_utils import *
from kinematics import get_transform_by_translation_and_theta, get_transform_by_r_and_theta
import numpy as np
import matplotlib.pyplot as plt
import pdb
from argparse import ArgumentParser
import logging

# create logger
logger = logging.getLogger('__name__')


def get_cme_parameters(params):
    se3s, timestamps = get_ground_truth_poses_from_csv(params.input_path)

    gt_x, gt_y, gt_th = get_x_y_th_from_se3s(se3s)
    cme_gt_x, cme_gt_y, cme_gt_th = [], [], []
    # Try and find a theta and curvature value that gets as close to the pose as possible
    for i in range(params.num_samples):
        idx = i
        logger.debug(se3s[idx])

        se3_gt = se3s[idx]
        th_gt = np.arctan2(se3_gt[1, 0], se3_gt[0, 0])
        x_gt = se3_gt[0, 3]
        y_gt = se3_gt[1, 3]

        r_estimate = x_gt / np.sin(th_gt)
        logger.debug(f'R, theta estimate: {r_estimate, th_gt}')

        se3_from_r_theta = get_transform_by_r_and_theta(r_estimate, th_gt)
        x_est = se3_from_r_theta[0, 3]
        y_est = se3_from_r_theta[1, 3]
        th_est = np.arctan2(se3_from_r_theta[1, 0], se3_from_r_theta[0, 0])
        cme_gt_x.append(x_est)
        cme_gt_y.append(y_est)
        cme_gt_th.append(th_est)
        logger.debug(f'GT: {x_gt}, {y_gt}, {th_gt}')
        logger.debug(f'Est: {x_est}, {y_est}, {th_est}')

    plt.figure(figsize=(15, 5))
    dim = params.num_samples + 50
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
    plt.savefig("%s%s" % (params.output_path, "/pose_comparison.pdf"))
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--output_path', type=str, default="",
                        help='Path to folder where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=100,
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
    get_cme_parameters(params)
    logger.info("Finished.")

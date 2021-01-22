# Script for finding motion between two sets of points using the lines that connect them
import numpy as np
import matplotlib.pyplot as plt
from get_rigid_body_motion import get_motion_estimate_from_svd
import pdb


def main():
    print("Running...")
    theta = -np.pi / 16
    T_offset = np.array([[0], [-1]])
    R_offset = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x_coords = np.array([-1, 0, 1, 1, -1])
    y_coords = np.array([1, 1.5, 1, -1, -1])
    P1 = np.array([x_coords, y_coords])
    P2 = R_offset @ P1 + T_offset

    # Recreate P1 and P2, now with an outlier each
    P1 = np.array([np.append(x_coords, 3), np.append(y_coords, 0)])
    P2 = np.array([np.append(x_coords, -3), np.append(y_coords, -1)])
    P2 = R_offset @ P2 + T_offset

    matched_points = []
    plt.figure(figsize=(10, 10))
    plt.plot(P1[0], P1[1], 'x')
    plt.plot(P2[0], P2[1], 'x')
    for idx in range(P1.shape[1]):
        x1 = P1[0, idx]
        y1 = P1[1, idx]
        x2 = P2[0, idx]
        y2 = P2[1, idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
        matched_points.append([x1, x2, y1, y2])

    plt.title("Proposed matches")
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    dim = 5
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("/workspace/Desktop/tmp.png")
    plt.close()

    # print(matched_points)

    # Mini RANSAC method
    pose_estimates = get_pose_estimates_with_ransac(P1, P2)
    print(pose_estimates)


def get_pose_estimates_with_ransac(P1, P2, iterations=6):
    # Mini RANSAC method
    pose_estimates = []
    for i in range(iterations):
        n_total = P1.shape[1]
        n_samples = 3
        # Select 3 matches at random
        subset_indices = np.random.choice(n_total, n_samples, replace=False)

        subset_P1 = []
        subset_P2 = []
        for idx in subset_indices:
            subset_P1.append([P1[0][idx], P1[1][idx]])
            subset_P2.append([P2[0][idx], P2[1][idx]])

        subset_P1 = np.transpose(subset_P1)
        subset_P2 = np.transpose(subset_P2)

        weights = np.ones(subset_P1.shape[1])
        v, theta_R = get_motion_estimate_from_svd(subset_P1, subset_P2, weights)
        pose = [v[0], v[1], theta_R]
        pose_estimates.append(pose)
    return pose_estimates


if __name__ == "__main__":
    main()

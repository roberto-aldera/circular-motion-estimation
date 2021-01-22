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
    # P1 = np.array([x_coords, y_coords])
    # P2 = R_offset @ P1 + T_offset

    # Recreate P1 and P2, now with an outlier each
    P1 = np.array([np.append(x_coords, 3), np.append(y_coords, 0)])
    P2 = np.array([np.append(x_coords, -3), np.append(y_coords, -1)])
    P2 = R_offset @ P2 + T_offset

    # Run SVD on all the points
    v, theta_R = get_motion_estimate_from_svd(P1, P2, weights=np.ones(P1.shape[1]))
    pose = [v[0], v[1], theta_R]
    T_model = np.transpose(np.array([pose[0:2]]))
    theta_model = pose[2]
    # pdb.set_trace()

    R_model = np.array([[np.cos(theta_model), -np.sin(theta_model)], [np.sin(theta_model), np.cos(theta_model)]])
    P_model = R_model @ P1 + T_model

    matched_points = []
    plt.figure(figsize=(10, 10))
    plt.plot(P1[0], P1[1], 'x', color="tab:blue")
    plt.plot(P2[0], P2[1], 'x', color="tab:orange")
    plt.plot(P_model[0], P_model[1], 'x', color="tab:green")
    for idx in range(P1.shape[1]):
        x1 = P1[0, idx]
        y1 = P1[1, idx]
        x2 = P2[0, idx]
        y2 = P2[1, idx]
        x3 = P_model[0, idx]
        y3 = P_model[1, idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
        plt.plot([x2, x3], [y2, y3], 'r--', linewidth=0.5, alpha=1)

        matched_points.append([x1, x2, y1, y2])

    plt.title("Proposed matches")
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    dim = 5
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("/workspace/Desktop/full_svd.png")
    plt.close()

    # print(matched_points)

    # Mini RANSAC method
    pose_estimates = get_pose_estimates_with_ransac(P1, P2)
    print(pose_estimates)

    # Transform points using estimate, and plot (green, hallucinated)
    model_x_y_th = pose_estimates[0]
    T_model = np.transpose(np.array([model_x_y_th[0:2]]))
    theta_model = model_x_y_th[2]
    # pdb.set_trace()

    R_model = np.array([[np.cos(theta_model), -np.sin(theta_model)], [np.sin(theta_model), np.cos(theta_model)]])
    P_model = R_model @ P1 + T_model

    plt.figure(figsize=(10, 10))
    plt.plot(P1[0], P1[1], 'x', color="tab:blue")
    plt.plot(P2[0], P2[1], 'x', color="tab:orange")
    plt.plot(P_model[0], P_model[1], 'x', color="tab:green")
    for idx in range(P1.shape[1]):
        x1 = P1[0, idx]
        y1 = P1[1, idx]
        x2 = P2[0, idx]
        y2 = P2[1, idx]
        x3 = P_model[0, idx]
        y3 = P_model[1, idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
        plt.plot([x2, x3], [y2, y3], 'r--', linewidth=0.5, alpha=1)

    plt.title("Proposed matches")
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    dim = 5
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("/workspace/Desktop/ransac_svd.png")
    plt.close()

    # TODO: Find errors (red lines)


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

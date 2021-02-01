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
    plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/figs_toy_ransac/full_svd.png")
    plt.close()

    # Mini RANSAC method
    pose_estimates = get_pose_estimates_with_ransac(P1, P2)
    best_model_index = get_best_ransac_motion_estimate_index(P1, P2, pose_estimates)

    model_x_y_th = pose_estimates[best_model_index]
    T_model = np.transpose(np.array([model_x_y_th[0:2]]))
    theta_model = model_x_y_th[2]

    R_model = np.array([[np.cos(theta_model), -np.sin(theta_model)], [np.sin(theta_model), np.cos(theta_model)]])
    P_model = R_model @ P1 + T_model
    plot_points_and_match_errors(P1, P2, P_model)


def get_best_ransac_motion_estimate_index(P1, P2, ransac_pose_estimates, inlier_threshold=0.1):
    # P1 an P2 are assumed to be already matched
    # Transform points using estimate, and plot (green, hallucinated) for all estimates
    model_inlier_counts = []
    for i in range(len(ransac_pose_estimates)):
        print("Computing RANSAC motion estimate for sample set:", i + 1, "of", len(ransac_pose_estimates))
        model_x_y_th = ransac_pose_estimates[i]
        T_model = np.transpose(np.array([model_x_y_th[0:2]]))
        theta_model = model_x_y_th[2]

        R_model = np.array([[np.cos(theta_model), -np.sin(theta_model)], [np.sin(theta_model), np.cos(theta_model)]])
        P_model = R_model @ P1 + T_model

        match_deltas = []
        for idx in range(P1.shape[1]):
            x2 = P2[0, idx]
            y2 = P2[1, idx]
            x3 = P_model[0, idx]
            y3 = P_model[1, idx]
            match_deltas.append([x2, x3, y2, y3])

        # Find errors (red lines) from match deltas
        match_error_magnitudes = []
        for match in match_deltas:
            x1 = match[0]
            x2 = match[1]
            y1 = match[2]
            y2 = match[3]
            match_error_magnitudes.append(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))
        # print("Match error magnitude for each point:", match_error_magnitudes)
        num_inliers = (np.array(match_error_magnitudes) < inlier_threshold).sum()

        model_inlier_counts.append(num_inliers)
    print("Inliers for each model:", model_inlier_counts)
    return np.argmax(model_inlier_counts)


def get_all_inliers_from_best_ransac_motion_estimate(P1, P2, ransac_pose_estimates, inlier_threshold=0.1):
    # P1 an P2 are assumed to be already matched
    # Transform points using estimate, and plot (green, hallucinated) for all estimates
    # Keep a record of the current inlier indices, and if they belong to the champion at the end, return them
    model_inlier_counts = []
    highest_inlier_count = 0
    champion_inliers = []
    for i in range(len(ransac_pose_estimates)):
        # print("Computing RANSAC motion estimate for sample set:", i + 1, "of", len(ransac_pose_estimates))
        model_x_y_th = ransac_pose_estimates[i]
        T_model = np.transpose(np.array([model_x_y_th[0:2]]))
        theta_model = model_x_y_th[2]

        R_model = np.array([[np.cos(theta_model), -np.sin(theta_model)], [np.sin(theta_model), np.cos(theta_model)]])
        P_model = R_model @ P1 + T_model

        match_deltas = []
        for idx in range(P1.shape[1]):
            x2 = P2[0, idx]
            y2 = P2[1, idx]
            x3 = P_model[0, idx]
            y3 = P_model[1, idx]
            match_deltas.append([x2, x3, y2, y3])

        # Find errors (red lines) from match deltas
        match_error_magnitudes = []
        for match in match_deltas:
            x1 = match[0]
            x2 = match[1]
            y1 = match[2]
            y2 = match[3]
            match_error_magnitudes.append(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))
        # print("Match error magnitude for each point:", match_error_magnitudes)
        num_inliers = (np.array(match_error_magnitudes) < inlier_threshold).sum()
        # check here if this is the best model so far, if it is, store the inliers
        if num_inliers > highest_inlier_count:
            champion_inliers = np.array(match_error_magnitudes) < inlier_threshold
            highest_inlier_count = num_inliers

        model_inlier_counts.append(num_inliers)
    print("Inliers for each model:", model_inlier_counts)
    print("Highest inlier count:", highest_inlier_count)
    return champion_inliers


def plot_points_and_match_errors(P1, P2, P_model,
                                 figpath="/workspace/data/landmark-distortion/ro_state_pb_developing/figs_toy_ransac/",
                                 figname="best_ransac_svd.pdf"):
    plt.figure(figsize=(10, 10))
    plt.plot(P1[0], P1[1], '+', markerfacecolor='none', markersize=1, color="tab:blue", label="Live")
    plt.plot(P2[0], P2[1], '+', markerfacecolor='none', markersize=1, color="tab:orange", label="Previous")
    plt.plot(P_model[0], P_model[1], '+', markerfacecolor='none', markersize=1, color="tab:green", label="Model")

    match_deltas = []
    for idx in range(P1.shape[1]):
        x1 = P1[0, idx]
        y1 = P1[1, idx]
        x2 = P2[0, idx]
        y2 = P2[1, idx]
        x3 = P_model[0, idx]
        y3 = P_model[1, idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
        plt.plot([x2, x3], [y2, y3], 'r', linewidth=0.5, alpha=1)
        match_deltas.append([x2, x3, y2, y3])

    plt.title("RANSAC proposed matches")
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    # dim = 5
    # plt.xlim(-dim, dim)
    # plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig("%s%s" % (figpath, figname))
    plt.close()


def get_best_ransac_motion_estimate_index_plot_all(P1, P2, ransac_pose_estimates, inlier_threshold=0.1,
                                                   figpath="/workspace/data/landmark-distortion/ro_state_pb_developing/figs_toy_ransac/"):
    # P1 an P2 are assumed to be already matched
    # Transform points using estimate, and plot (green, hallucinated) for all estimates
    model_inlier_counts = []
    for i in range(len(ransac_pose_estimates)):
        print("Computing RANSAC motion estimate for sample set:", i + 1, "of", len(ransac_pose_estimates))
        model_x_y_th = ransac_pose_estimates[i]
        T_model = np.transpose(np.array([model_x_y_th[0:2]]))
        theta_model = model_x_y_th[2]

        R_model = np.array([[np.cos(theta_model), -np.sin(theta_model)], [np.sin(theta_model), np.cos(theta_model)]])
        P_model = R_model @ P1 + T_model

        plt.figure(figsize=(10, 10))
        plt.plot(P1[0], P1[1], '+', markerfacecolor='none', markersize=1, color="tab:blue")
        plt.plot(P2[0], P2[1], '+', markerfacecolor='none', markersize=1, color="tab:orange")
        plt.plot(P_model[0], P_model[1], '+', markerfacecolor='none', markersize=1, color="tab:green")
        match_deltas = []
        for idx in range(P1.shape[1]):
            x1 = P1[0, idx]
            y1 = P1[1, idx]
            x2 = P2[0, idx]
            y2 = P2[1, idx]
            x3 = P_model[0, idx]
            y3 = P_model[1, idx]
            plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
            plt.plot([x2, x3], [y2, y3], 'r', linewidth=0.5, alpha=1)
            match_deltas.append([x2, x3, y2, y3])

        plt.title("RANSAC proposed matches")
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        # dim = 5
        # plt.xlim(-dim, dim)
        # plt.ylim(-dim, dim)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("%s%i%s" % (figpath, i, "_ransac_svd.pdf"))
        plt.close()

        # Find errors (red lines) from match deltas
        match_error_magnitudes = []
        for match in match_deltas:
            x1 = match[0]
            x2 = match[1]
            y1 = match[2]
            y2 = match[3]
            match_error_magnitudes.append(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))
        # print("Match error magnitude for each point:", match_error_magnitudes)
        num_inliers = (np.array(match_error_magnitudes) < inlier_threshold).sum()
        model_inlier_counts.append(num_inliers)
    print("Inliers for each model:", model_inlier_counts)
    return np.argmax(model_inlier_counts)


def get_pose_estimates_with_ransac(P1, P2, iterations=6):
    # RANSAC method
    # P1 an P2 are assumed to be already matched
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

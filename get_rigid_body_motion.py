# Script for finding rigid body motion with SVD given landmarks and their matches
import numpy as np
import matplotlib.pyplot as plt
import pdb


def get_motion_estimate_from_svd(P1, P2, weights):
    # Prepare cross-dispersion matrix C (...called S in RO code)
    # Find mean x, y of the two landmark sets
    P1_mean = np.mean(P1, axis=1)
    P2_mean = np.mean(P2, axis=1)

    # Transform points to bring them to the origin
    # This double transpose seems cumbersome and unnecessary
    P1_origin = np.transpose(np.transpose(P1) - P1_mean)
    P2_origin = np.transpose(np.transpose(P2) - P2_mean)

    # Weights (I think these can be used if we trust some points more than others)
    W = np.diag(weights)

    C = P1_origin @ W @ np.transpose(P2_origin)

    # Do SVD to find rotation
    # "A procedure for determining rigid body transformation parameters" - Challis 1995
    # "Least-Squares Fitting of Two 3-D Point Sets" - Arun 1987
    U, S, V = np.linalg.svd(C)
    M = np.identity(2)
    M[1, 1] = np.linalg.det(U @ np.transpose(V))
    R_without_M = U @ np.transpose(V)  # (as per Arun 1987, according to Challis 1995)
    R_final = U @ M @ np.transpose(V)
    # the reflection comes out backwards unless we used V*U'
    # -> there is an inconsistency between Challis 1995 and Arun 1987 I think
    R_final = np.transpose(R_final)
    theta_R = np.arctan2(R_final[1, 0], R_final[0, 0])

    # Find transform vector v
    v = P2_mean - R_final @ P1_mean
    return v, theta_R


def main():
    print("Running motion estimation from points using toy data...")
    N = 5
    theta = -np.pi / 8
    T_offset = np.array([[-2], [-3.5]])
    R_offset = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    P1 = np.transpose(np.array([[2, 0], [1, 0], [0, 0], [0, 0.5], [0, 1]]))
    P2 = R_offset @ P1 + T_offset

    weights = np.ones(N)
    v, theta_R = get_motion_estimate_from_svd(P1, P2, weights)
    print("True translation:", T_offset, "\n estimate:", v)
    print("True rotation:", theta, "\n estimate:", theta_R)

    plt.figure(figsize=(10, 10))
    plt.plot(P1[0], P1[1], 'x')
    plt.plot(P2[0], P2[1], 'o')
    plt.title("Title")
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("/workspace/Desktop/tmp.png")
    plt.close()


if __name__ == "__main__":
    main()

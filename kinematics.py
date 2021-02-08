import numpy as np
import matplotlib.pyplot as plt
import pdb


def main():
    print("Running kinematics on toy data...")
    theta = 0  # np.pi / 32
    T_offset = np.array([[0], [1]])

    pose = np.identity(4)
    pose[0, 0] = np.cos(theta)
    pose[0, 1] = -np.sin(theta)
    pose[1, 0] = np.sin(theta)
    pose[1, 1] = np.cos(theta)
    pose[0, 3] = T_offset[0]
    pose[1, 3] = T_offset[1]

    x_coords = np.array([-1, 0, 1, 1, -1])
    y_coords = np.array([1, 1.5, 1, -1, -1])
    num_points = len(x_coords)
    P1 = np.array([x_coords, y_coords, np.zeros(num_points), np.ones(num_points)])
    P2 = np.linalg.inv(pose) @ P1

    start_position = np.array([0, 0])
    end_position = [0, 0]
    end_position = np.r_[end_position, 0, 1]  # add z = 0, and final 1 for homogenous coordinates for se3 multiplication
    end_position = pose @ end_position

    print("start_position:", start_position)
    print("end_position:", end_position)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(start_position[0], start_position[1], 'ro')
    plt.plot(end_position[0], end_position[1], 'g*')
    plt.plot(P1[0], P1[1], 'rx')

    for idx in range(num_points):
        x1 = start_position[0]
        y1 = start_position[1]
        x2 = P1[0][idx]
        y2 = P1[1][idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)

    plt.title("Frame 1")
    plt.grid()
    dim = 5
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(1, 2, 2)
    plt.plot(P2[0], P2[1], 'gx')
    relative_end_position = np.linalg.inv(pose) @ end_position
    plt.plot(relative_end_position[0], relative_end_position[1], 'g*')

    for idx in range(num_points):
        x1 = relative_end_position[0]
        y1 = relative_end_position[1]
        x2 = P2[0][idx]
        y2 = P2[1][idx]
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)

    plt.title("Frame 2")
    plt.grid()
    dim = 5
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/toy_kinematics.png")
    plt.close()


if __name__ == "__main__":
    main()

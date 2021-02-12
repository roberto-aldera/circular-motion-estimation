import numpy as np
import matplotlib.pyplot as plt
import pdb
from kinematics import get_transform_by_translation_and_theta, get_transform_by_r_and_theta, \
    get_robot_to_world_transform


def make_plot(landmark_1, start_position, end_position, radius, range_from_pose_2, estimated_radius):
    T_robot_to_world = get_robot_to_world_transform()
    end_position = T_robot_to_world @ end_position
    landmark_1 = T_robot_to_world @ landmark_1

    plt.figure(figsize=(10, 10))
    plt.plot(start_position[0], start_position[1], 'ro')
    plt.plot(end_position[0], end_position[1], 'g*')
    plt.plot(landmark_1[0], landmark_1[1], 'r^')

    plt.plot([start_position[0], landmark_1[0][0]], [start_position[1], landmark_1[1][0]], 'k', linewidth=0.5, alpha=1)
    plt.plot([end_position[0], landmark_1[0][0]], [end_position[1], landmark_1[1][0]], 'k', linewidth=0.5, alpha=1)

    # Draw motion circle
    circle_theta = np.linspace(0, 2 * np.pi, 50)
    x1 = radius + radius * np.cos(circle_theta)
    x2 = radius * np.sin(circle_theta)
    plt.plot(x1, x2, 'k--')

    # Draw estimated motion circle
    x1 = estimated_radius + estimated_radius * np.cos(circle_theta)
    x2 = estimated_radius * np.sin(circle_theta)
    # plt.plot(x1, x2, 'k--')

    # Draw landmark range
    x1 = landmark_1[0] + range_from_pose_2 * np.cos(circle_theta)
    x2 = landmark_1[1] + range_from_pose_2 * np.sin(circle_theta)
    plt.plot(x1, x2, 'r-.')

    plt.title("Circular motion world")
    plt.grid()
    # dim = 5
    # plt.xlim(-dim, dim)
    # plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/r_and_theta.png")
    plt.close()


def main():
    print("Running R and theta on toy data...")
    k_radius = 5
    k_theta = np.pi / 8
    pose = get_transform_by_r_and_theta(rotation_radius=k_radius, theta=k_theta)
    x_coords = np.array([5])
    y_coords = np.array([-2])
    num_points = len(x_coords)
    landmark_1 = np.array([x_coords, y_coords, np.zeros(num_points), np.ones(num_points)])
    landmark_2 = np.linalg.inv(pose) @ landmark_1

    start_position = np.array([0, 0])
    end_position = [0, 0]
    end_position = np.r_[end_position, 0, 1]  # add z = 0, and final 1 for homogenous coordinates for se3 multiplication
    end_position = pose @ end_position

    print("start_position:", start_position)
    print("end_position:", end_position)

    # Get landmark position relative to start position in range and angle
    range_1 = np.linalg.norm(np.transpose(landmark_1[0:2]))
    range_2 = np.linalg.norm(np.transpose(landmark_2[0:2]))
    print("Range 1:", range_1)
    print("Range 2:", range_2)

    # Get relative angles
    angle_1 = np.arctan2((landmark_1[1]), (landmark_1[0]))
    angle_2 = np.arctan2((landmark_2[1]), (landmark_2[0]))
    print("Angle 1:", angle_1 * 180 / np.pi)
    print("Angle 2:", angle_2 * 180 / np.pi)

    #############
    # d_1 = 10
    # d_2 = 7
    # phi_1 = -np.pi / 4
    # phi_2 = -np.pi / 2.1  # a bit more than 90 deg to the left
    d_1 = range_1
    d_2 = range_2
    phi_1 = angle_1
    phi_2 = angle_2

    theta = 2 * np.arctan(
        (np.sin(np.pi + phi_2) - (d_1 / d_2) * np.sin(-phi_1)) / ((d_1 / d_2) * np.cos(-phi_1) - np.cos(np.pi + phi_2)))
    radius = (d_2 * np.sin(phi_1 - phi_2 - theta)) / (2 * np.sin(theta / 2) * np.sin(-phi_1 + (theta / 2)))
    print("Actual R and theta:", k_radius, ",", k_theta * 180 / np.pi)
    print("Estimated R and theta:", radius, ",", theta * 180 / np.pi)

    make_plot(landmark_1, start_position, end_position, k_radius, range_2, estimated_radius=radius)


if __name__ == "__main__":
    main()

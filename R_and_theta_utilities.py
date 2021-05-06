import numpy as np
import matplotlib.pyplot as plt
import pdb
import csv
from kinematics import get_transform_by_translation_and_theta, get_transform_by_r_and_theta, \
    get_robot_to_world_transform


def make_plot(landmarks, start_position, end_position, landmark_ranges, radius=None, estimated_radii=None):
    T_robot_to_world = get_robot_to_world_transform()
    end_position = T_robot_to_world @ end_position

    plt.figure(figsize=(10, 10))
    plt.plot(start_position[0], start_position[1], 'ro')
    plt.plot(end_position[0], end_position[1], 'g*')

    # Draw motion circle
    circle_theta = np.linspace(0, 2 * np.pi, 500)
    if radius:
        x1 = radius + radius * np.cos(circle_theta)
        x2 = radius * np.sin(circle_theta)
        plt.plot(x1, x2, 'k--')

    for idx in range(landmarks.shape[1]):
        landmark_1 = T_robot_to_world @ landmarks[:, idx].reshape(-1, 1)
        # landmark_1 = landmarks[:, idx].reshape(-1, 1)

        plt.plot(landmark_1[0], landmark_1[1], 'r^')

        plt.plot([start_position[0], landmark_1[0][0]], [start_position[1], landmark_1[1][0]], 'k', linewidth=0.5,
                 alpha=1)
        plt.plot([end_position[0], landmark_1[0][0]], [end_position[1], landmark_1[1][0]], 'k', linewidth=0.5, alpha=1)

        # Draw landmark range
        x1 = landmark_1[0] + landmark_ranges[idx] * np.cos(circle_theta)
        x2 = landmark_1[1] + landmark_ranges[idx] * np.sin(circle_theta)
        plt.plot(x1, x2, 'r-.')

    # Draw estimated motion circle
    if estimated_radii:
        for estimated_radius in estimated_radii:
            x1 = estimated_radius + estimated_radius * np.cos(circle_theta)
            x2 = estimated_radius * np.sin(circle_theta)
            plt.plot(x1, x2, 'k--')

    plt.title("Circular motion world")
    plt.grid()
    dim = 10
    plt.xlim(-dim, dim)
    plt.ylim(-dim, dim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("/workspace/data/landmark-distortion/ro_state_pb_developing/r_and_theta.png")
    plt.close()


def get_relative_range_and_bearing_from_x_and_y(relative_x, relative_y):
    # Get landmark position relative to robot position in range and angle
    relative_range = np.linalg.norm(np.array([relative_x, relative_y]))
    relative_angle = np.arctan2(relative_y, relative_x)
    return relative_range, relative_angle


def get_theta_and_radius_from_single_match(d_1, d_2, phi_1, phi_2):
    if d_1 == d_2 and phi_1 == phi_2:
        # TODO: come back and debug this, not sure why this case occurs but it does sometimes
        # theta = np.nan
        # radius = np.nan
        theta = 0
        radius = np.inf
        # print("duplicated occurred...")
        # print("d_1:", d_1, "d_2:", d_2)
        # print("phi_1:", phi_1, "phi_2:", phi_2)

    else:
        theta = 2 * np.arctan(
            (-np.sin(phi_2) + (d_1 / d_2) * np.sin(phi_1)) / ((d_1 / d_2) * np.cos(phi_1) + np.cos(phi_2)))
        radius = (d_2 * np.sin(phi_1 - phi_2 - theta)) / (2 * np.sin(theta / 2) * np.sin(-phi_1 + (theta / 2)))
    return theta, radius


def get_theta_and_curvature_from_single_match(d_1, d_2, phi_1, phi_2):
    if d_1 == d_2 and phi_1 == phi_2:
        # This can occur when landmarks are in the exact same position as their match (stationary vehicle).
        theta = 0
        radius = np.inf

    else:
        theta = 2 * np.arctan(
            (-np.sin(phi_2) + (d_1 / d_2) * np.sin(phi_1)) / ((d_1 / d_2) * np.cos(phi_1) + np.cos(phi_2)))
        radius = (d_2 * np.sin(phi_1 - phi_2 - theta)) / (2 * np.sin(theta / 2) * np.sin(-phi_1 + (theta / 2)))

    # Handle possibility of radius being zero gracefully
    if radius == 0.0:
        curvature = np.inf
    else:
        curvature = 1 / radius

    return theta, curvature


def debugging_from_csv_points():
    print("Running R and theta on debugging data (real landmarks)...")
    csv_file = "/workspace/data/landmark-distortion/ro_state_pb_developing/points.csv"
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        points_data = list(reader)
    points_data = points_data[:5]  # just first few points while I'm debugging.
    x1s = [float(item[2]) for item in points_data]
    y1s = [float(item[3]) for item in points_data]
    x2s = [float(item[0]) for item in points_data]
    y2s = [float(item[1]) for item in points_data]
    x_coords_1 = np.array(x1s)
    y_coords_1 = np.array(y1s)
    x_coords_2 = np.array(x2s)
    y_coords_2 = np.array(y2s)

    num_points = len(x_coords_1)
    landmarks_1 = np.array([x_coords_1, y_coords_1, np.zeros(num_points), np.ones(num_points)])
    landmarks_2 = np.array([x_coords_2, y_coords_2, np.zeros(num_points), np.ones(num_points)])

    landmark_ranges = []
    estimated_radii = []

    for idx in range(num_points):
        landmark_1 = landmarks_1[:, idx].reshape(-1, 1)
        landmark_2 = landmarks_2[:, idx].reshape(-1, 1)

        # Get landmark position relative to robot position in range and angle
        range_1, angle_1 = get_relative_range_and_bearing_from_x_and_y(landmark_1[0], landmark_1[1])
        range_2, angle_2 = get_relative_range_and_bearing_from_x_and_y(landmark_2[0], landmark_2[1])
        print("Range 1:", range_1)
        print("Range 2:", range_2)
        print("Angle 1:", angle_1 * 180 / np.pi)
        print("Angle 2:", angle_2 * 180 / np.pi)

        landmark_ranges.append(range_2)

        theta, radius = get_theta_and_radius_from_single_match(d_1=range_1, d_2=range_2, phi_1=angle_1, phi_2=angle_2)
        print("Estimated R and theta:", radius, ",", theta * 180 / np.pi)
        estimated_radii.append(radius)

        circular_motion_pose = get_transform_by_r_and_theta(radius, theta)
        print("x:", circular_motion_pose[0, 3])
        print("y:", circular_motion_pose[1, 3])
        end_position = [circular_motion_pose[0, 3], circular_motion_pose[1, 3]]

    start_position = np.array([0, 0])
    # end_position = [0, 0]
    end_position = np.r_[end_position, 0, 1]
    make_plot(landmarks_1, start_position, end_position, landmark_ranges, estimated_radii=estimated_radii)


def multiple_landmarks_test():
    print("Running R and theta on toy data for multiple landmarks...")
    k_radius = 10
    k_theta = np.pi / 8
    pose = get_transform_by_r_and_theta(rotation_radius=k_radius, theta=k_theta)
    x_coords = np.array([5, 4])
    y_coords = np.array([-2, -1])
    start_position = np.array([0, 0])
    end_position = [0, 0]
    end_position = np.r_[end_position, 0, 1]  # add z = 0, and final 1 for homogenous coordinates for se3 multiplication
    end_position = pose @ end_position
    print("start_position:", start_position)
    print("end_position:", end_position)

    num_points = len(x_coords)
    landmarks = np.array([x_coords, y_coords, np.zeros(num_points), np.ones(num_points)])
    landmark_ranges = []

    for idx in range(num_points):
        landmark_1 = landmarks[:, idx].reshape(-1, 1)
        landmark_2 = np.linalg.inv(pose) @ landmark_1

        # Get landmark position relative to robot position in range and angle
        range_1, angle_1 = get_relative_range_and_bearing_from_x_and_y(landmark_1[0], landmark_1[1])
        range_2, angle_2 = get_relative_range_and_bearing_from_x_and_y(landmark_2[0], landmark_2[1])
        print("Range 1:", range_1)
        print("Range 2:", range_2)
        print("Angle 1:", angle_1 * 180 / np.pi)
        print("Angle 2:", angle_2 * 180 / np.pi)

        landmark_ranges.append(range_2)

        theta, radius = get_theta_and_radius_from_single_match(d_1=range_1, d_2=range_2, phi_1=angle_1, phi_2=angle_2)
        print("Actual R and theta:", k_radius, ",", k_theta * 180 / np.pi)
        print("Estimated R and theta:", radius, ",", theta * 180 / np.pi)

        circular_motion_pose = get_transform_by_r_and_theta(radius, theta)
        print("x:", circular_motion_pose[0, 3])
        print("y:", circular_motion_pose[1, 3])

    make_plot(landmarks, start_position, end_position, landmark_ranges, k_radius)


def single_landmark_test():
    print("Running R and theta on toy data...")
    k_radius = 10
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

    landmark_ranges = []

    # Get landmark position relative to robot position in range and angle
    range_1, angle_1 = get_relative_range_and_bearing_from_x_and_y(landmark_1[0], landmark_1[1])
    range_2, angle_2 = get_relative_range_and_bearing_from_x_and_y(landmark_2[0], landmark_2[1])
    # range_1, angle_1 = get_relative_range_and_bearing_from_x_and_y(-2.24, -6.22)
    # range_2, angle_2 = get_relative_range_and_bearing_from_x_and_y(-0.1, -6.20)
    # range_1, angle_1 = get_relative_range_and_bearing_from_x_and_y(-2.82, -6.24)
    # range_2, angle_2 = get_relative_range_and_bearing_from_x_and_y(-0.69, -6.20)
    print("Range 1:", range_1)
    print("Range 2:", range_2)
    print("Angle 1:", angle_1 * 180 / np.pi)
    print("Angle 2:", angle_2 * 180 / np.pi)
    landmark_ranges.append(range_2)

    theta, radius = get_theta_and_radius_from_single_match(d_1=range_1, d_2=range_2, phi_1=angle_1, phi_2=angle_2)
    print("Actual R and theta:", k_radius, ",", k_theta * 180 / np.pi)
    print("Estimated R and theta:", radius, ",", theta * 180 / np.pi)

    circular_motion_pose = get_transform_by_r_and_theta(radius, theta)
    print("x:", circular_motion_pose[0, 3])
    print("y:", circular_motion_pose[1, 3])

    make_plot(landmark_1, start_position, end_position, landmark_ranges, k_radius)


def handcrafted_landmark_test():
    print("Running R and theta on handcrafted data...")
    # Get landmark position relative to robot position in range and angle
    range_1, angle_1 = get_relative_range_and_bearing_from_x_and_y(0, 1)
    range_2, angle_2 = get_relative_range_and_bearing_from_x_and_y(0, 1.1)

    print("Range 1:", range_1)
    print("Range 2:", range_2)
    print("Angle 1:", angle_1 * 180 / np.pi)
    print("Angle 2:", angle_2 * 180 / np.pi)

    theta, radius = get_theta_and_radius_from_single_match(d_1=range_1, d_2=range_2, phi_1=angle_1, phi_2=angle_2)
    print("Estimated R and theta:", radius, ",", theta * 180 / np.pi)

    circular_motion_pose = get_transform_by_r_and_theta(radius, theta)
    print("x:", circular_motion_pose[0, 3])
    print("y:", circular_motion_pose[1, 3])


if __name__ == "__main__":
    # single_landmark_test()
    handcrafted_landmark_test()
    # multiple_landmarks_test()
    # debugging_from_csv_points()

import numpy as np
import sys
import os
import csv


def get_poses_from_file(dataset_path):
    """
    Load poses from monolithic file.
    """
    print('Loading poses from monolithic file...')

    # Magic
    sys.path.append(os.path.expanduser("/workspace/code/corelibs/src/tools-python"))
    sys.path.append(os.path.expanduser("/workspace/code/corelibs/build/datatypes"))
    sys.path.append(os.path.expanduser("/workspace/code/corelibs/build/datatypes/datatypes_python"))

    from mrg.logging import MonolithicDecoder
    from mrg.adaptors.transform import PbSerialisedTransformToPython

    # Open monolithic and iterate frames
    relative_poses_path = dataset_path
    print("reading relative_poses_path: " + relative_poses_path)
    monolithic_decoder = MonolithicDecoder(
        relative_poses_path)

    # iterate mono
    se3s = []
    timestamps = []
    for pb_serialised_transform, _, _ in monolithic_decoder:
        # adapt
        serialised_transform = PbSerialisedTransformToPython(
            pb_serialised_transform)
        se3s.append(serialised_transform[0])
        timestamps.append(serialised_transform[1])

    print("Finished reading", len(timestamps), "poses.")
    return se3s, timestamps


def get_x_y_th_velocities_from_poses(se3s, timestamps):
    assert len(se3s) == len(timestamps)
    x_velocities = []
    y_velocities = []
    th_velocities = []
    for i in range(len(timestamps) - 1):
        delta_time = timestamps[i + 1] / 1e6 \
                     - timestamps[i] / 1e6
        se3 = se3s[i]
        x_pose = se3[0, -1]
        x_velocity = x_pose / delta_time
        x_velocities.append(x_velocity)
        y_pose = se3[1, -1]
        y_velocity = y_pose / delta_time
        y_velocities.append(y_velocity)
        th_pose = np.arctan2(se3[1, 0], se3[0, 0])
        th_velocity = th_pose / delta_time
        th_velocities.append(th_velocity)
    return x_velocities, y_velocities, th_velocities


def get_speeds(se3s, timestamps):
    assert len(se3s) == len(timestamps)
    speeds = []
    for i in range(len(timestamps) - 1):
        delta_time = timestamps[i + 1] / 1e6 \
                     - timestamps[i] / 1e6
        se3 = se3s[i]
        translation = se3[0:2, -1]
        incremental_distance = np.linalg.norm(translation)
        speed = incremental_distance / delta_time
        speeds.append(speed)
    return speeds


def get_speeds_from_x_y_th(x_y_th, timestamps):
    assert len(x_y_th) == len(timestamps)
    speeds = []
    x_values = [item[0] for item in x_y_th]
    y_values = [item[1] for item in x_y_th]
    for i in range(len(timestamps) - 1):
        delta_time = timestamps[i + 1] / 1e6 \
                     - timestamps[i] / 1e6
        translation = [x_values[i], y_values[i]]
        incremental_distance = np.linalg.norm(translation)
        speed = incremental_distance / delta_time
        speeds.append(speed)
    return speeds


def get_poses(se3s):
    poses = []
    pose = np.identity(4)
    for i in range(len(se3s)):
        pose = pose * se3s[i]
        poses.append(pose)

    x_position = [pose[0, 3] for pose in poses]
    y_position = [pose[1, 3] for pose in poses]
    return x_position, y_position


def save_poses_to_csv(se3_poses, timestamps, pose_source, export_folder):
    # Save poses with format: timestamp, dx, dy, dth
    with open("%s%s%s" % (export_folder, pose_source, "_poses.csv"), 'w') as poses_file:
        wr = csv.writer(poses_file, delimiter=",")
        for idx in range(len(se3_poses)):
            timestamp_and_pose_estimate = [timestamps[idx], se3_poses[idx][0, 3], se3_poses[idx][1, 3],
                                           np.arctan2(se3_poses[idx][1, 0], se3_poses[idx][0, 0])]
            wr.writerow(timestamp_and_pose_estimate)


def get_timestamps_and_x_y_th_from_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        pose_data = list(reader)

    timestamps = [int(item[0]) for item in pose_data]
    x_y_th = [items[1:] for items in pose_data]
    return timestamps, x_y_th


def save_timestamps_and_x_y_th_to_csv(timestamps, x_y_th, pose_source, export_folder):
    # Save poses with format: timestamp, dx, dy, dth
    with open("%s%s%s" % (export_folder, pose_source, "_poses.csv"), 'w') as poses_file:
        wr = csv.writer(poses_file, delimiter=",")
        x_values = [item[0] for item in x_y_th]
        y_values = [item[1] for item in x_y_th]
        th_values = [item[2] for item in x_y_th]
        for idx in range(len(timestamps)):
            timestamp_and_x_y_th = [timestamps[idx], x_values[idx], y_values[idx], th_values[idx]]
            wr.writerow(timestamp_and_x_y_th)

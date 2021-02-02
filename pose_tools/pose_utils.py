import numpy as np
import sys
import os
import csv
import pandas as pd
from liegroups import SE3


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


def get_poses_from_serialised_transform(pb_serialised_transform):
    # Magic
    sys.path.append(os.path.expanduser("/workspace/code/corelibs/src/tools-python"))
    sys.path.append(os.path.expanduser("/workspace/code/corelibs/build/datatypes"))
    sys.path.append(os.path.expanduser("/workspace/code/corelibs/build/datatypes/datatypes_python"))

    # from mrg.logging import MonolithicDecoder
    from mrg.adaptors.transform import PbSerialisedTransformToPython

    serialised_transform = PbSerialisedTransformToPython(pb_serialised_transform)
    se3 = serialised_transform[0]
    timestamp = serialised_transform[1]
    return se3, timestamp


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


def get_x_y_th_velocities_from_x_y_th(x_y_th, timestamps):
    assert len(x_y_th) == len(timestamps)
    x_velocities = []
    y_velocities = []
    th_velocities = []
    for i in range(len(timestamps) - 1):
        delta_time = timestamps[i + 1] / 1e6 \
                     - timestamps[i] / 1e6
        x_y_th_sample = np.array(x_y_th[i])
        # import pdb
        # pdb.set_trace()
        x_pose = float(x_y_th_sample[0])
        x_velocity = x_pose / delta_time
        x_velocities.append(x_velocity)
        y_pose = float(x_y_th_sample[1])
        y_velocity = y_pose / delta_time
        y_velocities.append(y_velocity)
        th_pose = float(x_y_th_sample[2])
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
        pose = pose @ se3s[i]
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


def get_ground_truth_poses_from_csv(path_to_gt_csv):
    """
    Load poses from csv for the Oxford radar robotcar 10k dataset.
    """
    df = pd.read_csv(path_to_gt_csv)
    # print(df.head())
    x_vals = df['x']
    y_vals = df['y']
    th_vals = df['yaw']
    timestamps = df['source_radar_timestamp']

    se3s = []
    for i in range(len(df.index)):
        th = -th_vals[i]
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = x_vals[i]
        pose[1, 3] = y_vals[i]
        se3s.append(pose)
    return se3s, timestamps


def get_ground_truth_poses_from_csv_as_se3(path_to_gt_csv):
    """
    Load poses from csv for the Oxford radar robotcar 10k dataset as SE3s.
    """
    df = pd.read_csv(path_to_gt_csv)
    # print(df.head())
    x_vals = df['x']
    y_vals = df['y']
    th_vals = df['yaw']
    timestamps = df['source_radar_timestamp']

    se3s = []
    for i in range(len(df.index)):
        th = th_vals[i]
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = x_vals[i]
        pose[1, 3] = y_vals[i]
        se3s.append(SE3.from_matrix(np.asarray(pose)))
    return se3s, timestamps


def get_se3s_from_raw_se3s(raw_se3s):
    """
    Transform from raw se3 matrices into fancier SE3 type
    """
    se3s = []
    for pose in raw_se3s:
        se3s.append(SE3.from_matrix(np.asarray(pose)))
    return se3s


def get_se3s_from_x_y_th(x_y_th):
    """
    Transform from x_y_th list into fancier SE3 type
    """
    se3s = []
    for sample in x_y_th:
        th = float(sample[2])
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = float(sample[0])
        pose[1, 3] = float(sample[1])
        se3s.append(SE3.from_matrix(np.asarray(pose)))
    return se3s


def get_raw_se3s_from_x_y_th(x_y_th):
    """
    Transform from x_y_th list into raw SE3 type
    """
    se3s = []
    for sample in x_y_th:
        th = float(sample[2])
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = float(sample[0])
        pose[1, 3] = float(sample[1])
        se3s.append(pose)
    return se3s


if __name__ == "__main__":
    print("Running post utils main...")
    get_ground_truth_poses_from_csv(
        "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/gt/radar_odometry.csv")

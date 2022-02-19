import sys
import os
import numpy as np
from argparse import ArgumentParser

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/Users/roberto/code/corelibs/src/tools-python")
sys.path.insert(-1, "/Users/roberto/code/corelibs/build/datatypes")
sys.path.insert(-1, "/Users/roberto/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


class RadarOdometryState(object):

    def __init__(self):
        """Creates an empty radar odometry state with no parameters set.
        """
        self.timestamp = None
        self.primary_scan_landmark_set = None
        self.secondary_scan_landmark_set = None
        self.primary_scan_point_descriptors = None
        self.secondary_scan_point_descriptors = None
        self.unary_match_candidates = None
        self.associations = None
        self.compatibility_matrix = None
        self.eigen_vector = None
        self.eigen_values = None
        self.selected_matches = None
        self.number_of_matches = None
        self.motion_estimate = None
        self.g_motion_estimate = None
        self.xyz_yaw_covariance = None
        self.raw_transforms = None
        self.match_ratio = None


class Matrix(object):

    def __init__(self):
        """Creates an empty matrix with no parameters set.
        """
        self.rows = None
        self.cols = None
        self.float_data = None
        self.double_data = None
        self.int32_data = None
        self.int64_data = None


def get_ro_state_from_pb(pb_ro_state):
    radar_odometry_state = RadarOdometryState()
    radar_odometry_state.timestamp = pb_ro_state.timestamp
    radar_odometry_state.primary_scan_landmark_set = pb_ro_state.primary_scan_landmark_set
    radar_odometry_state.secondary_scan_landmark_set = pb_ro_state.secondary_scan_landmark_set
    radar_odometry_state.primary_scan_point_descriptors = pb_ro_state.primary_scan_point_descriptors
    radar_odometry_state.secondary_scan_point_descriptors = pb_ro_state.secondary_scan_point_descriptors
    radar_odometry_state.unary_match_candidates = pb_ro_state.unary_match_candidates
    radar_odometry_state.associations = pb_ro_state.associations
    radar_odometry_state.compatibility_matrix = pb_ro_state.compatibility_matrix
    radar_odometry_state.eigen_vector = pb_ro_state.eigen_vector
    radar_odometry_state.eigen_values = pb_ro_state.eigen_values
    radar_odometry_state.selected_matches = pb_ro_state.selected_matches
    radar_odometry_state.number_of_matches = pb_ro_state.number_of_matches
    radar_odometry_state.motion_estimate = pb_ro_state.motion_estimate
    radar_odometry_state.g_motion_estimate = pb_ro_state.g_motion_estimate
    radar_odometry_state.xyz_yaw_covariance = pb_ro_state.xyz_yaw_covariance
    radar_odometry_state.raw_transforms = pb_ro_state.raw_transforms
    radar_odometry_state.match_ratio = pb_ro_state.match_ratio
    return radar_odometry_state


def get_matrix_from_pb(pb_matrix, stored_datatype='double'):
    # This function is a special case where we know we are wanting to deserialise a pbMatrix, and that the stored data
    # is in doubles. If we need other data types later, we can expand from here
    matrix = Matrix()
    matrix.rows = pb_matrix.rows
    matrix.cols = pb_matrix.cols
    # matrix.float_data = pb_matrix.float_data
    matrix.double_data = pb_matrix.double_data
    # matrix.int32_data = pb_matrix.int32_data
    # matrix.int64_data = pb_matrix.int64_data
    assert stored_datatype == 'double'
    reshaped_matrix = np.reshape(np.array([matrix.double_data]), (matrix.rows, matrix.cols))
    return reshaped_matrix


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--ro_state_mono', type=str, default="", help='Path to radar odometry state monolithic')
    params = parser.parse_args()

    print("reading ro_state_path: " + params.ro_state_mono)

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.ro_state_mono)
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))
    idx = 0
    pb_state, name_scan, _ = radar_state_mono[idx]
    ro_state = get_ro_state_from_pb(pb_state)
    print("Processing index:", idx)
    print("Timestamp:", ro_state.timestamp)

    # Get a landmarks set (requires handling point clouds) as an example
    primary_scan_landmark_set = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set)
    print("Number of landmarks:", primary_scan_landmark_set.num_points)
    print("Landmarks xyz data:", primary_scan_landmark_set.get_xyz())

    # Get unary matches as an example
    unary_match_candidates = get_matrix_from_pb(ro_state.unary_match_candidates)
    unary_matches = np.reshape(unary_matches, (unary_matches.shape[1], -1))
    print("Shape of unary matches:", unary_match_candidates.shape)
    print(unary_match_candidates)

    # Get associations (if they're stored)
    associations = get_matrix_from_pb(ro_state.associations)
    associations = np.reshape(get_matrix_from_pb(ro_state.associations), (associations.shape[1], -1))
    print("Shape of associations:", associations.shape)
    print(associations)

    print("Finished!")

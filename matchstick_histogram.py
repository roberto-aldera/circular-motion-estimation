import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
from pose_tools.pose_utils import *
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud


def matchstick_histogram(params, radar_state_mono):
    figure_path = params.input_path + "figs_matchstick_histogram/"
    output_path = Path(figure_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    for i in range(params.num_samples):
        plt.figure(figsize=(20, 20))
        # dim = 200
        # plt.xlim(-dim, dim)
        # plt.ylim(-dim, dim)
        pb_state, name_scan, _ = radar_state_mono[i]
        ro_state = get_ro_state_from_pb(pb_state)
        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        primary_landmarks = np.c_[
            primary_landmarks, np.ones(len(primary_landmarks))]  # so that se3 multiplication works

        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))

        print("Size of primary landmarks:", len(primary_landmarks))
        print("Size of secondary landmarks:", len(secondary_landmarks))

        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        matches_to_plot = selected_matches.astype(int)

        print("Processing index: ", i)
        # plot x and y swapped around so that robot is moving forward as upward direction
        plt.plot(primary_landmarks[:, 1], primary_landmarks[:, 0], '+', markerfacecolor='none', markersize=1,
                 color="tab:blue")
        plt.plot(secondary_landmarks[:, 1], secondary_landmarks[:, 0], '+', markerfacecolor='none',
                 markersize=1, color="tab:orange")
        matched_points = []
        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5, alpha=1)
            matched_points.append([x1, x2, y1, y2])

        # plot sensor range for Oxford radar robotcar dataset
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        r = 163
        circle_x1 = 0 + r * np.cos(circle_theta)
        circle_x2 = 0 + r * np.sin(circle_theta)
        plt.plot(circle_x1, circle_x2, 'g--')

        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("%s%s%i%s" % (output_path, "/landmark_matches_", i, ".pdf"))
        plt.close()

        distances = []
        angles = []
        for match in matched_points:
            x1 = match[0]
            x2 = match[1]
            y1 = match[2]
            y2 = match[3]
            distances.append(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))
            angles.append(np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi)

        plt.figure(figsize=(10, 10))
        plt.plot(distances, '.')
        plt.title("Matchstick length")
        plt.grid()
        plt.xlabel("Match index (in order of preference)")
        plt.ylabel("Distance between matched points (m)")
        plt.savefig("%s%s%i%s" % (output_path, "/matchstick_length_", i, ".png"))
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(angles, '.')
        plt.title("Matchstick angle")
        plt.grid()
        plt.xlabel("Match index (in order of preference)")
        plt.ylabel("Angle between match and horizontal (deg)")
        plt.savefig("%s%s%i%s" % (output_path, "/matchstick_angles_", i, ".png"))
        plt.close()

        plt.figure(figsize=(10, 10))
        n, bins, patches = plt.hist(distances, 100, density=False, facecolor='tab:blue')
        plt.title("Distribution of matchstick lengths")
        plt.grid()
        plt.xlabel("Length (m)")
        plt.ylabel("Number of matches")
        plt.savefig("%s%s%i%s" % (output_path, "/matchstick_length_histogram_", i, ".png"))
        plt.close()

        plt.figure(figsize=(10, 10))
        n, bins, patches = plt.hist(angles, 100, density=False, facecolor='tab:blue')
        plt.title("Distribution of matchstick angles")
        plt.grid()
        plt.xlabel("Angle (deg)")
        plt.ylabel("Number of matches")
        plt.savefig("%s%s%i%s" % (output_path, "/matchstick_angle_histogram_", i, ".png"))
        plt.close()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--num_samples', type=int, default=settings.TOTAL_SAMPLES,
                        help='Number of samples to process')
    params = parser.parse_args()

    print("Running matchstick histogram script...")

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    print("Number of indices in this radar odometry state monolithic:", len(radar_state_mono))

    # get a landmark set in and plot it
    matchstick_histogram(params, radar_state_mono)


if __name__ == "__main__":
    main()

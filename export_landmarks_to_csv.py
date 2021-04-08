# Get landmarks from an RO state monolithic and store them in a csv file for a network's dataloader

import numpy as np
import matplotlib.pyplot as plt
import pdb
from argparse import ArgumentParser
import logging
from pathlib import Path
import shutil
import sys
import csv
from tqdm import tqdm
from unpack_ro_protobuf import get_ro_state_from_pb, get_matrix_from_pb

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.pointcloud import PbSerialisedPointCloudToPython
from mrg.pointclouds.classes import PointCloud

# create logger
logger = logging.getLogger('__name__')


def get_landmarks_as_matches(params, radar_state_mono, save_to_csv=True):
    timestamps_from_ro_state = []
    
    export_path = params.output_path + "exported_matched_landmarks/"
    output_path = Path(export_path)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    for i in tqdm(range(params.num_samples)):
        pb_state, name_scan, _ = radar_state_mono[i]
        ro_state = get_ro_state_from_pb(pb_state)
        timestamps_from_ro_state.append(ro_state.timestamp)

        primary_landmarks = PbSerialisedPointCloudToPython(ro_state.primary_scan_landmark_set).get_xyz()
        secondary_landmarks = PbSerialisedPointCloudToPython(ro_state.secondary_scan_landmark_set).get_xyz()
        selected_matches = get_matrix_from_pb(ro_state.selected_matches).astype(int)
        selected_matches = np.reshape(selected_matches, (selected_matches.shape[1], -1))

        logger.debug(f'Size of primary landmarks {len(primary_landmarks)}')
        logger.debug(f'Size of secondary landmarks: {len(secondary_landmarks)}')

        # Selected matches are those that were used by RO, best matches are for development purposes here in python land
        matches_to_plot = selected_matches.astype(int)

        logger.debug(f'Processing index: {i}')
        matched_points = []

        for match_idx in range(len(matches_to_plot)):
            x1 = primary_landmarks[matches_to_plot[match_idx, 1], 1]
            y1 = primary_landmarks[matches_to_plot[match_idx, 1], 0]
            x2 = secondary_landmarks[matches_to_plot[match_idx, 0], 1]
            y2 = secondary_landmarks[matches_to_plot[match_idx, 0], 0]
            matched_points.append([x1, x2, y1, y2])

        if save_to_csv:
            save_matched_landmarks_to_csv(matched_points, i, output_path)


def save_matched_landmarks_to_csv(matched_landmarks, idx, export_folder):
    with open("%s%s%i%s" % (export_folder, "/matched_landmarks_", idx, ".csv"), 'w') as poses_file:
        wr = csv.writer(poses_file, delimiter=",")
        for i in range(len(matched_landmarks)):
            wr.writerow(matched_landmarks[i])


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--input_path', type=str, default="",
                        help='Path to folder containing required inputs')
    parser.add_argument('--output_path', type=str, default="",
                        help='Path to folder where outputs will be saved')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to process')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Logging level')
    params = parser.parse_args()

    logging_level = logging.DEBUG if params.verbose > 0 else logging.INFO
    logger.setLevel(logging_level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logger.info("Running script...")

    # --input_path /workspace/data/landmark-distortion/ro_state_pb_developing/ro_state_files/
    # --output_path /workspace/data/landmark-distortion/cme_ground_truth/
    # --num_samples 6900

    # You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
    # -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
    radar_state_mono = IndexedMonolithic(params.input_path + "ro_state.monolithic")
    logger.info(f'Number of indices in this radar odometry state monolithic: {len(radar_state_mono)}')

    get_landmarks_as_matches(params, radar_state_mono, save_to_csv=True)

    logger.info("Finished.")

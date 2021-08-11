# Project-specific settings live here.
import numpy as np
import random

# Ensure reproducibility
np.random.seed(0)
random.seed(0)

# Colours
GT_COLOUR = "tab:green"
RO_COLOUR = "tab:blue"
AUX1_COLOUR = "tab:red"

# Offset
K_RADAR_INDEX_OFFSET = 0

# Aux names
AUX0_NAME = "ransac-inliers-svd"
AUX1_NAME = "cm-iqr-thetas"
AUX2_NAME = "cm-iqr-curvature"
AUX3_NAME = "cm-iqr-both"
AUX4_NAME = "cm-kde"
AUX5_NAME = "aux5"

# General dataset parameters
TOTAL_SAMPLES = 10000
RADAR_IMAGE_DIMENSION = 3600

# RO state path
RUNNING_ON_SERVER = False

if RUNNING_ON_SERVER:
    RO_STATE_PATH = "/Volumes/scratchdata/roberto/ro_state_files/"
    POSE_OUTPUT_PATH = "/Volumes/scratchdata/roberto/pose-outputs/"
else:
    RO_STATE_PATH = "/workspace/data/ro-state-files/radar_oxford_10k/2019-01-10-14-50-05/"
    POSE_OUTPUT_PATH = RO_STATE_PATH
    # POSE_OUTPUT_PATH = "/workspace/data/landmark-distortion/RANSAC-baseline/pose-outputs/"

# Landmark paths
ROOT_DATA_DIR = "/workspace/data/landmark-distortion/"
RO_LANDMARKS_DIR = ROOT_DATA_DIR
OUTPUT_EXPORT_DIR = RO_LANDMARKS_DIR  # "/workspace/data/radar-tmp/"
FIGS_DIR = OUTPUT_EXPORT_DIR + "figs/"

# Radar scan paths
ROOT_DIR = "/workspace/data/landmark-distortion/"
RADAR_IMAGE_DIR = ROOT_DIR + "radar-images/"
# RADAR_DATASET_PATH = "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/radar/" \
#                      "cts350x_103517/2019-01-10-14-50-10"

## These paths are for generating figures to show concepts
# RADAR_DATASET_PATH = "/Volumes/mrgdatastore6/Logs/Roofus/2018-03-09-14-01-57-thorsmork-boulders-3-small-radar-bins/" \
#                      "logs/radar/cts350x/2018-03-09-14-02-03"
# RADAR_DATASET_PATH = "/Volumes/mrgdatastore6/Logs/Roofus/2018-03-09-13-49-07-thorsmork-boulders-2/" \
#                      "logs/radar/cts350x/2018-03-09-13-49-13"
# RADAR_DATASET_PATH = "/Volumes/mrgdatastore5/Logs/Muttley/2018-05-02-15-02-32-oxford-10k-laser3d-radar/" \
#                      "logs/radar/cts350x/2018-05-02-14-02-36"
# RADAR_DATASET_PATH = "/Volumes/mrgdatastore6/Logs/Muttley/2019-01-10-11-46-21-radar-oxford-10k/" \
#                      "logs/radar/cts350x_103517/2019-01-10-11-46-26"
RADAR_DATASET_PATH = "/Volumes/mrgdatastore6/Logs/Penfold/2020-01-31-14-46-47-medium-sax/" \
                     "logs/radar/cts350x_102717/2020-01-31-14-46-54"

RAW_SCAN_MONOLITHIC = RADAR_DATASET_PATH + "/cts350x_raw_scan.monolithic"
RADAR_CONFIG = RADAR_DATASET_PATH + "/cts350x_config.monolithic"

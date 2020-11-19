# Project-specific settings live here.
import numpy as np
import random

# Ensure reproducibility
np.random.seed(0)
random.seed(0)

# General dataset parameters
TOTAL_SAMPLES = 1
RADAR_IMAGE_DIMENSION = 3000

# Landmark paths
ROOT_DATA_DIR = "/workspace/data/landmark-distortion/"
RO_LANDMARKS_DIR = ROOT_DATA_DIR
OUTPUT_EXPORT_DIR = RO_LANDMARKS_DIR  # "/workspace/data/radar-tmp/"
FIGS_DIR = OUTPUT_EXPORT_DIR + "figs/"

# Radar scan paths
ROOT_DIR = "/workspace/data/landmark-distortion/"
RADAR_IMAGE_DIR = ROOT_DIR + "radar-images/"
RADAR_DATASET_PATH = "/workspace/data/RadarDataLogs/2019-01-10-14-50-05-radar-oxford-10k/radar/" \
                     "cts350x_103517/2019-01-10-14-50-10"
RAW_SCAN_MONOLITHIC = RADAR_DATASET_PATH + "/cts350x_raw_scan.monolithic"
RADAR_CONFIG = RADAR_DATASET_PATH + "/cts350x_config.monolithic"

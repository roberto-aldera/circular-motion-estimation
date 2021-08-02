import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import time
from pathlib import Path
import shutil
from argparse import ArgumentParser
import settings
import pdb

sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-utilities/build/radarutilities_datatypes_python")
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python/mrg/adaptors")
from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.radar import pbNavtechRawConfigToPython, pbNavtechRawScanToPython


def get_radar_image(params, radar_mono, config, split_data_path, idx):
    subset_start_index = params.subset_start_index
    image_dimension = params.image_dimension
    sensor_rotation = params.rotation_angle
    output_file_extension = params.output_file_extension
    intensity_multiplier = 2

    scan_index = subset_start_index + idx
    pb_raw_scan, name_scan, _ = radar_mono[scan_index]
    radar_sweep = pbNavtechRawScanToPython(pb_raw_scan, config)

    width, height, res = (image_dimension,
                          image_dimension,
                          config.bin_size_or_resolution)
    cart_img = radar_sweep.GetCartesian(pixel_width=width, pixel_height=height, resolution=res,
                                        method='cv2', verbose=False)
    img = Image.fromarray(cart_img.astype(np.uint8) * intensity_multiplier, 'L')
    img = img.rotate(sensor_rotation)
    img = ImageOps.mirror(img)
    img_as_np_array = np.array(img)

    img.save("%s%s%i%s" % (split_data_path, "/", scan_index, output_file_extension))
    img.close()
    print("Generated samples up to index:", scan_index, "with dim =", image_dimension,
          "and written to:", split_data_path)

    # return img_as_np_array
    return cart_img


def overlay_landmarks_onto_radar_image(params, radar_img, config, output_path, idx):
    landmarks_csv_path = params.landmarks_path + "landmarks_" + str(idx)
    print(landmarks_csv_path)
    landmarks = np.genfromtxt(landmarks_csv_path + ".csv", delimiter=",")
    landmarks[:, 0], landmarks[:, 1] = np.array(landmarks[:, 1]), np.array(landmarks[:, 0])

    # print(landmarks)
    # th = 0.9 + np.pi  # TODO - get this from platform config
    # rotation_matrix = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # rotate
    # landmarks = landmarks @ rotation_matrix
    # mirror (z-axis is down for robot)
    # landmarks[:, 0] = -landmarks[:, 0]
    # translate (platform config is accounted for in RO, so need to move landmarks back to radar frame)
    # print(metric_scale_factor)
    # landmarks[:, 0] = landmarks[:, 0] - 0.05 * metric_scale_factor
    # landmarks[:, 1] = landmarks[:, 1] - 0.0526 * metric_scale_factor
    # landmarks[:, 0] = landmarks[:, 0] - 0.8
    # landmarks[:, 1] = landmarks[:, 1] - 1.2

    # plt.figure(figsize=(20, 20))
    # dim = 200
    # plt.xlim(-dim, dim)
    # plt.ylim(-dim, dim)
    # plt.plot(landmarks[:, 0], landmarks[:, 1], "*")
    # plt.grid()
    # plt.savefig("%s%s%i%s" % (output_path, "/only_landmarks_", idx, ".png"))
    # plt.close()

    plt.figure(figsize=(15, 15))
    plt.imshow(radar_img, cmap='gray')
    metric_scale_factor = 1 / config.bin_size_or_resolution
    image_landmarks = np.array(landmarks) * metric_scale_factor
    image_landmarks[:, 0], image_landmarks[:, 1] = image_landmarks[:, 0] + settings.RADAR_IMAGE_DIMENSION / 2, \
                                                   image_landmarks[:, 1] + settings.RADAR_IMAGE_DIMENSION / 2
    plt.scatter(image_landmarks[:, 0], image_landmarks[:, 1], marker='^', s=40, facecolors='none', edgecolors='r')
    dim = settings.RADAR_IMAGE_DIMENSION
    plt.xlim(0, dim)
    plt.ylim(0, dim)
    plt.savefig("%s%s%i%s" % (output_path, "/landmarks_", idx, ".jpg"))


def main():
    # Define a main loop to run and show some example data if this script is run as main
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--subset_start_index', type=int, default=0, help='Scan index from which to begin processing')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples for processing')
    parser.add_argument('--image_dimension', type=int, default=settings.RADAR_IMAGE_DIMENSION,
                        help='Exported image dimension')
    parser.add_argument('--rotation_angle', type=int, default=0, help='Account for sensor offset angle')
    parser.add_argument('--output_file_extension', type=str, default=".jpg", help='File extension for output images')
    parser.add_argument('--landmarks_path', type=str, default="",
                        help='Path to landmarks that were exported for processing')

    params = parser.parse_args() 
    print("Starting dataset generation...")
    start_time = time.time()

    print("Generating data, size =", params.num_samples)

    split_data_path = Path(settings.RADAR_IMAGE_DIR)
    if split_data_path.exists() and split_data_path.is_dir():
        shutil.rmtree(split_data_path)
    split_data_path.mkdir(parents=True)

    radar_config_mono = IndexedMonolithic(settings.RADAR_CONFIG)
    config_pb, name, timestamp = radar_config_mono[0]
    config = pbNavtechRawConfigToPython(config_pb)
    radar_mono = IndexedMonolithic(settings.RAW_SCAN_MONOLITHIC)

    # landmarks_path = params.landmarks_path
    # figure_path = landmarks_path + "figs/"
    # output_path = Path(figure_path)
    # if output_path.exists() and output_path.is_dir():
    #     shutil.rmtree(output_path)
    # output_path.mkdir(parents=True)

    for i in range(params.num_samples):
        radar_img = get_radar_image(params, radar_mono, config, split_data_path, i)
        # overlay_landmarks_onto_radar_image(params, radar_img, config, output_path, i)
    print("--- Radar image generation execution time: %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()

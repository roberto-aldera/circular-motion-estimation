import glob
from PIL import Image
import os
from argparse import ArgumentParser
from natsort import natsort_keygen, ns
import settings


def generate_thumbnails(params):
    thumbnail_dimension = 1024  # 512
    size = thumbnail_dimension, thumbnail_dimension
    images_batch = params.figs_path + "*.jpg"

    for infile in glob.glob(images_batch):
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        outfile = os.path.splitext(infile)[0] + ".thumbnail"
        im.save(outfile, "JPEG")


def generate_gif(params):
    fp_in = params.figs_path + "*.thumbnail"
    fp_out = params.figs_path + "landmarks.gif"
    natural_sort_key = natsort_keygen(alg=ns.IGNORECASE)  # do a "human sort" for proper input file ordering
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=natural_sort_key)]

    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=250, loop=0)

    # clean up thumbnails
    for item in os.listdir(params.figs_path):
        if item.endswith(".thumbnail"):
            os.remove(os.path.join(params.figs_path, item))


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--figs_path', type=str, default="",
                        help='Path to figures to make GIF from')
    params = parser.parse_args()

    generate_thumbnails(params)
    generate_gif(params)

import json
import numpy as np

from external_lib.pyflow import pyflow
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import cv2


def main(args):
    sequence_path = Path(args.sequence_path)
    out_path = sequence_path / "vdo_of"
    out_path.mkdir(exist_ok=True, parents=True)
    images = [x for x in (sequence_path / "vdo_frames").glob("*.jpg")]

    post_img = cv2.imread(
        str(sequence_path / "vdo_frames" / f"{1:05d}.jpg"),
        cv2.IMREAD_GRAYSCALE
    )
    im_shape = post_img.shape
    post_img = cv2.resize(post_img, (int(im_shape[0] / 8), int(im_shape[1] / 8)))

    for ii in range(2, len(images) + 1):
        prev_img = post_img
        post_img = cv2.imread(
            str(sequence_path / "vdo_frames" / f"{ii:05d}.jpg"),
            cv2.IMREAD_GRAYSCALE
        )
        im_shape = post_img.shape
        post_img = cv2.resize(post_img, (int(im_shape[0] / 8), int(im_shape[1] / 8)))

        img1 = np.atleast_3d(prev_img.astype(float) / 255.)
        img2 = np.atleast_3d(post_img.astype(float) / 255.)

        u, v, im2W = pyflow.coarse2fine_flow(img1, img2, 0.012, 0.75, 20, 7, 1, 30, 1)
        opflow = np.dstack((u, v))
        opflow = cv2.resize(opflow, im_shape)
        np.save(str(out_path / f"{ii - 1:05d}.npy"), opflow)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate optical flow files for each input image pair",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sequence_path",
        type=str,
        help="Path to vdo_frames root folder",
    )

    args = parser.parse_args()
    main(args)

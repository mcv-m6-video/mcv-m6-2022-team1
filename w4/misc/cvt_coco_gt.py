from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import csv
import json
import pandas as pd

from PIL import Image


def main(args):
    in_path = Path(args.gt_path)
    print(f"Converting stuff from {str(in_path)} into coco format...")
    with open(in_path, 'r') as f:
        anns = pd.read_csv(
            f,
            names=["frame", "id", "bb_left", "bb_top", "bb_width",
                     "bb_height", "conf", "x", "y", "z"]
        )

    with Image.open(str(in_path.parent.parent / "vdo_frames" / "00001.jpg")) as im:
        width, height = im.size

    info = {
        "year": 2022,
        "version": "0.0.0",
        "description": "haha",
        "contributor": "none",
        "url": "localhost",
        "date_created": None,
    }
    images = [{
        "id": ii,
        "width": width,
        "height": height,
        "file_name": f"{ii:05d}.jpg",
        "license": 1,
    } for ii in range(1, args.nframes + 1)]
    licenses = [{
        "id": 1,
        "name": "A",
        "url": "localhost",
    }]

    annotations = []
    id_anns = 1
    for ii in range(1, args.nframes + 1):
        for _, _, left, top, width, height, conf, _, _, _ in anns[anns["frame"] == ii].itertuples(index=False):
            if conf:
                annotations.append({
                    "id": id_anns,
                    "image_id": ii,
                    "category_id": 1,
                    "bbox": (left, top, width, height),
                    "area": width * height,
                    "iscrowd": 0,
                })
            id_anns += 1
    categories = [{
        "id": 1,
        "name": "car",
        "supercategory": "moving",
    }]

    cocodict = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "licenses": licenses,
        "categories": categories
    }
    with open(in_path.parent / "gt_coco.json", 'w') as f:
        json.dump(cocodict, f)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert a MOTS txt track into a COCO-compliant JSON file",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gt_path",
        type=str,
        help="Path to the ground truth data file"
    )
    parser.add_argument(
        "nframes",
        type=int,
        help="Number of frames in the video sequence"
    )
    args = parser.parse_args()
    main(args)

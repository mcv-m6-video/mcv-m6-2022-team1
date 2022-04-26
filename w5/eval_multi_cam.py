import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import motmetrics as mm

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from tqdm.auto import tqdm
from utils.eval import iou


def load_annotations(path: str) -> pd.DataFrame:
    """
    Loads a csv-like annotation file with fields ["frame", "ID", "left", "top",
    "width", "height", "confidence", "null1", "null2", "null3"] into a pandas
    dataframe. Check Nvidia AICity challenge readme for further detail.

    Parameters
    ----------
    path: str
        Path string for the input file.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe which contains the loaded csv with only the needed
        columns ["frame", "ID", "left", "top", "width", "height", "confidence"].
    """
    ann = pd.read_csv(
        path,
        sep=",",
        names=["frame", "ID", "left", "top", "width", "height", "confidence", "x", "y", "z"]
    )
    return ann


def generate_gt_from_xml(in_path: str) -> pd.DataFrame:
    dataset = ET.parse(in_path).getroot()

    # Separate annotations by frames. We do not care about the classes for the time
    # being, we only grab cars

    labels = ["frame", "ID", "left", "top", "width", "height", "confidence", "x", "y", "z"]
    annotations = []

    # [frame, ID, left, top, width, height, 1, -1, -1, -1]
    for track in dataset.findall("track"):
        if track.attrib["label"] == "car":
            for box in track.findall("box"):
                annotations.append([
                    int(box.attrib["frame"]),
                    int(track.attrib["id"]),
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]) - float(box.attrib["xtl"]),
                    float(box.attrib["ybr"]) - float(box.attrib["ytl"]),
                    1, -1, -1, -1
                ])
    return pd.DataFrame(annotations, columns=labels)


def main(args):
    out_path = Path(args.out_path)
    start_frame = args.start_frame
    end_frame = args.end_frame
    sequences = ["S01"]
    root_path = "/home/cisu/PycharmProjects/mcv-m6-2022-team1/w5/DATA/train/S01"
    root_path = Path(root_path)

    acc_list_views = []
    view_names = []
    for camera in root_path.glob("*"):
        name = str(camera).split("/")[-1]
        view_names.append(name)
        print(f"{name} in progress")
        gt_path = camera / "gt" / "gt.txt"
        pd_path = camera / "mtsc" / "mtsc_deepsort_mask_rcnn.txt"

        # Load data
        if gt_path.suffix == ".xml":
            gt_data = generate_gt_from_xml(str(gt_path))
        elif gt_path.suffix == ".txt":
            gt_data = load_annotations(str(gt_path))
        else:
            raise TypeError("Input file is neither a MOT .txt nor a MOT .xml")

        ann_data = load_annotations(str(pd_path))

        # 〈camera_id〉 〈obj_id〉 〈frame_id〉 〈xmin〉 〈ymin〉 〈width〉 〈height〉 〈-1〉 〈-1〉
        # Ground truth ids are the ones we want for reference
        unique_imgs_gt = gt_data["frame"].unique()

        unique_imgs_gt = unique_imgs_gt[
                np.logical_and(start_frame <= unique_imgs_gt, unique_imgs_gt <= end_frame)]

        unique_imgs_gt.sort()

        acc = mm.MOTAccumulator(auto_id=True)

        for ii in tqdm(unique_imgs_gt, desc="Evaluation progress..."):
            gt_current = gt_data[gt_data["frame"] == ii]
            ann_current = ann_data[ann_data["frame"] == ii]

                # Sanity check, although it should not be necessary
            if not len(gt_current):
                continue

            gt_ids = gt_current["ID"]
            if not len(ann_current):
                acc.update(
                    np.asarray(gt_ids),
                    np.asarray([]),
                    np.asarray([])
                )
                continue
            ann_ids = ann_current["ID"]

            gt_bboxes = np.asarray(gt_current[["left", "top", "width", "height"]])
            ann_bboxes = np.asarray(ann_current[["left", "top", "width", "height"]])

            # Convert to XYXY format
            gt_bboxes[:, 2] = gt_bboxes[:, 0] + gt_bboxes[:, 2]
            gt_bboxes[:, 3] = gt_bboxes[:, 1] + gt_bboxes[:, 3]
            ann_bboxes[:, 2] = ann_bboxes[:, 0] + ann_bboxes[:, 2]
            ann_bboxes[:, 3] = ann_bboxes[:, 1] + ann_bboxes[:, 3]

            intersect = iou(gt_bboxes, ann_bboxes)

            acc.update(
                np.asarray(gt_ids),
                np.asarray(ann_ids),
                1 - intersect
            )
        acc_list_views.append(acc)
    # Just print them, nothing better thus far
    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc_list_views,
        metrics=["idf1", "idp", "idr", "precision", "recall"],
        names=view_names)
    summary.to_csv(str(out_path))
    print(summary)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluate a track in MOT format",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gt_path",
        type=str,
        help="Path to the ground truth data. Can either be a MOT track or an"
             "xml file such as the one in the provided data",
    )
    parser.add_argument(
        "pd_path",
        type=str,
        help="Path to the prediction file. A MOT track in txt format",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Path to the output summary file (fn included)",
    )
    parser.add_argument(
        "start_frame",
        type=int,
        help="Starting frame to consider for testing",
    )
    parser.add_argument(
        "end_frame",
        type=int,
        help="Final frame to consider for testing",
    )

    args = parser.parse_args()
    main(args)

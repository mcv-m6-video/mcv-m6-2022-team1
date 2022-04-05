import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path


def draw_bboxes(img, gt_ann, pd_ann):
    for ann in gt_ann:
        bbox = ann["bbox"]
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0] + bbox[2]),
             int(bbox[1] + bbox[3])),
            color=(255, 0, 0),
            thickness=4
        )
        cv2.putText(
            img,
            f"Class: {ann['category_id']}",
            (int(bbox[0]), int(bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    for ann in pd_ann:
        bbox = ann["bbox"]
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0] + bbox[2]),
             int(bbox[1] + bbox[3])),
            color=(0, 255, 0),
            thickness=4
        )
        cv2.putText(
            img,
            f"Prediction: {ann['category_id']}",
            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3] + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    return img


def main(args):
    gt_path = Path(args.gt_path)
    pd_path = Path(args.pd_path)
    img_path = Path(args.img_path)
    out_path = Path(args.out_path)
    track_path = Path(args.track_path) if args.track_path is not None else None

    start_frame = args.start_frame
    end_frame = args.end_frame

    with open(gt_path, 'r') as f_gt:
        gt_dict = json.load(f_gt)

    gt_list = {ii["id"]: [] for ii in gt_dict["images"]}

    classes = {
        x["id"]: x["name"] for x in gt_dict["categories"]
    }

    for ann in gt_dict["annotations"]:
        gt_list[ann["image_id"]].append(ann)

    with open(pd_path, 'r') as f_pd:
        pd_dict = json.load(f_pd)

    pd_list = {ii["id"]: [] for ii in gt_dict["images"]}
    for ann in pd_dict:
        pd_list[ann["image_id"]].append(ann)

    if track_path is not None:
        pass

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(out_path), fourcc, 20.0, (1920, 1080))

    print(f"Output path: {str(out_path)}")

    for img_id in range(start_frame, end_frame + 1):
        img = cv2.imread(str(img_path / f"{img_id:05}.jpg"))
        img = draw_bboxes(img, gt_list[img_id], pd_list[img_id])
        img = cv2.resize(img, (1920, 1080))
        video.write(img)
    video.release()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create a video with bounding boxes from detections",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gt_path",
        type=str,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "img_path",
        type=str,
        help="Path to the image data",
    )
    parser.add_argument(
        "pd_path",
        type=str,
        help="Path to the image data",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Path to store output video",
    )
    parser.add_argument(
        "start_frame",
        type=int,
        help="Starting frame to plot",
    )
    parser.add_argument(
        "end_frame",
        type=int,
        help="End frame to plot",
    )
    parser.add_argument(
        "--track_path",
        type=str,
        help="End frame to plot",
        required=False,
        default=None
    )

    args = parser.parse_args()
    main(args)
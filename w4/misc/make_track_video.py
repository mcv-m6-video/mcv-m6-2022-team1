import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import pandas as pd


def draw_bboxes(img, gt_ann, pd_ann):
    for _, idt, left, top, width, height, _, _, _, _ in gt_ann.itertuples(index=False):
        cv2.rectangle(
            img,
            (int(left), int(top)),
            (int(left + width),
             int(top + height)),
            color=(255, 0, 0),
            thickness=4
        )
        cv2.putText(
            img,
            f"GT Track ID: {idt}",
            (int(left), int(top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    for _, idt, left, top, width, height, _, _, _, _ in pd_ann.itertuples(index=False):
        cv2.rectangle(
            img,
            (int(left), int(top)),
            (int(left + width),
             int(top + height)),
            color=(0, 255, 0),
            thickness=4
        )
        cv2.putText(
            img,
            f"Track ID: {idt}",
            (int(left + width), int(top + height + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    return img


def draw_trails(img, gt_trail, pd_trail):
    for trail in gt_trail:
        trail = trail.sort_values("frame")[["left", "top", "width", "height"]].to_numpy()
        trail[:, 0:2] += (trail[:, 2:] // 2)
        trail = trail[:, 0:2]
        cv2.polylines(
            img,
            [trail.astype(np.int32)],
            isClosed=False,
            color=(255, 0, 0),
            lineType=cv2.LINE_8,
            thickness=4
        )

    for trail in pd_trail:
        trail = trail.sort_values("frame")[["left", "top", "width", "height"]].to_numpy()
        trail[:, 0:2] += (trail[:, 2:] // 2)
        trail = trail[:, 0:2]
        cv2.polylines(
            img,
            [trail.astype(np.int32)],
            isClosed=False,
            color=(0, 255, 0),
            lineType=cv2.LINE_8,
            thickness=4
        )


def main(args):
    gt_path = Path(args.gt_path)
    pd_path = Path(args.pd_path)
    img_path = Path(args.img_path)
    out_path = Path(args.out_path)

    start_frame = args.start_frame
    end_frame = args.end_frame

    gt = pd.read_csv(
        str(gt_path),
        names=["frame", "ID", "left", "top", "width",
               "height", "confidence", "x", "y", "z"]
    )
    pred = pd.read_csv(
        str(pd_path),
        names=["frame", "ID", "left", "top", "width",
               "height", "confidence", "x", "y", "z"]
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(out_path), fourcc, 20.0, (1920, 1080))

    print(f"Output path: {str(out_path)}")

    for img_id in range(start_frame, end_frame + 1):

        frame_gt = gt[gt["frame"] == img_id]
        frame_pd = pred[pred["frame"] == img_id]

        gt_track_ids = frame_gt["ID"].unique()
        pd_track_ids = frame_pd["ID"].unique()

        gt_trails = [gt[np.logical_and(gt["ID"] == idt, gt["frame"] <= img_id)]
                     for idt in gt_track_ids]
        pd_trails = [pred[np.logical_and(pred["ID"] == idt, pred["frame"] <= img_id)]
                     for idt in pd_track_ids]

        img = cv2.imread(str(img_path / f"{img_id:05}.jpg"))
        draw_trails(img, gt_trails, pd_trails)
        img = draw_bboxes(
            img,
            frame_gt,
            frame_pd
        )
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

    args = parser.parse_args()
    main(args)
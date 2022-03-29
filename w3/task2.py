import json
from tqdm import tqdm
from track import Track
import numpy as np


def iou(gt: list(), pred: list()):
    gt[2], gt[3] = gt[0] + gt[2], gt[1] + gt[3]
    pred[2], pred[3] = pred[0] + pred[2], pred[1] + pred[3]

    xA = max(gt[0], pred[0])
    yA = max(gt[1], pred[1])
    xB = min(gt[2], pred[2])
    yB = min(gt[3], pred[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def read_detections(json_file: str) -> list():
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def track_max_overlap(data, init_frame_id, last_frame_id, IoU_threshold=0.2):
    # Assumes first frame as initialization

    tracking_list = list()  # list of Track objects
    track_id = 0

    for frame_id in tqdm(range(init_frame_id, last_frame_id + 1)):
        frame_detections = [x for x in data if x["image_id"] == frame_id]

        num_detections = len(frame_detections)
        idx_assigned = [-1] * num_detections

        for ii, object_in_frame in enumerate(frame_detections):
            if frame_id == init_frame_id:  # initial frame
                # appends new track
                new_track = Track(track_id)
                new_track.append_bbox(object_in_frame["bbox"])
                new_track.append_frame_id_appearence(frame_id)
                tracking_list.append(new_track)
                track_id += 1

            else:
                # check IoU of every detected object_in_frame in new frame with previous detected bboxes in
                # tracking_list
                for track_prev in tracking_list:
                    if track_prev.frame_id_appearence[-1] == frame_id - 1: # if appeared in last frame, go on
                        iou_between_currentNprev = iou(object_in_frame["bbox"], track_prev.bbox[-1])
                        if iou_between_currentNprev > IoU_threshold:
                            track_prev.append_bbox(object_in_frame["bbox"])  # updates new bbox position
                            idx_assigned[ii] = 0 # this object was assigned
                            track_prev.append_frame_id_appearence(frame_id)

        # unassigned new objects -> new track
        if frame_id != init_frame_id:
            for jj, is_assigned in enumerate(idx_assigned):
                if is_assigned == -1:  # if not assigned
                    new_track = Track(track_id)
                    new_track.append_bbox(frame_detections[jj]["bbox"])
                    new_track.append_frame_id_appearence(frame_id)
                    tracking_list.append(new_track)
                    track_id += 1


    return tracking_list

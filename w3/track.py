import json
import csv
import numpy as np
from tqdm import tqdm
import cv2


class Track(object):
    def __init__(self, id: int):
        self.id = id  # track id
        self.bbox = list()  # list of bbox associated to an object
        self.frame_id_appearence = list()

    def append_bbox(self, new_bbox):
        self.bbox.append(new_bbox)

    def append_frame_id_appearence(self, frame_id_):
        self.frame_id_appearence.append(frame_id_)


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
                    if not track_prev.frame_id_appearence[-1] == frame_id:  # ignore if already updated
                        if track_prev.frame_id_appearence[-1] == frame_id - 1:  # if appeared in last frame, go on
                            iou_between_currentNprev = iou(object_in_frame["bbox"], track_prev.bbox[-1])
                            # FIXME: use higher IoU instead of just this
                            if iou_between_currentNprev > IoU_threshold:
                                track_prev.append_bbox(object_in_frame["bbox"])  # updates new bbox position
                                idx_assigned[ii] = 0  # this object was assigned
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


def iou(gt: list, pred: list):
    gt_x1, gt_y1, gt_w, gt_h = gt
    pd_x1, pd_y1, pd_w, pd_h = pred

    gt_x2 = gt_x1 + gt_w
    pd_x2 = pd_x1 + pd_w

    gt_y2 = gt_y1 + gt_h
    pd_y2 = pd_y1 + pd_h

    xA = max(gt_x1, pd_x1)
    yA = max(gt_y1, pd_y1)
    xB = min(gt_x2, pd_x2)
    yB = min(gt_y2, pd_y2)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
    boxBArea = (pd_x2 - pd_x1 + 1) * (pd_y2 - pd_y1 + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def visualize_overlap(track_list, frame_loader, num_of_colors=200):
    color_list = [tuple(np.random.choice(range(256), size=3)) for ic in range(num_of_colors)]

    for img_frame_id, img in tqdm(frame_loader):
        img = np.array(img)
        bboxes_to_draw = list()

        for track in track_list:
            # if track has stored the frame id, extract bbox and append to draw it
            try:
                index = track.frame_id_appearence.index(img_frame_id)
                associated_id = track.id
                bboxes_to_draw.append((track.bbox[index], associated_id))
            except ValueError:
                pass
        for (x, y, w, h), id_track in bboxes_to_draw:
            cv2.circle(img, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), -1)
            cv2.putText(img, f"id: {id_track}", (int(x + (w / 2)), int(y + (h / 2))), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (int(color_list[id_track % num_of_colors][0]), int(color_list[id_track % num_of_colors][1]),
                         int(color_list[id_track % num_of_colors][2])), 2, cv2.LINE_AA)
            # cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255,0,0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('', img)
        cv2.waitKey(0)


def eval_file(track_list, init_frame_id, last_frame_id, csv_file):
    for frame_id in tqdm(range(init_frame_id, last_frame_id + 1)):
        for track in track_list:
            try:
                index = track.frame_id_appearence.index(frame_id)
                id_csv = track.id
                bbox = track.bbox[index]
                csv_row = [str(frame_id), str(id_csv), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), "-1", "-1", "-1", "-1"]
                with open(csv_file, 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(csv_row)
            except ValueError:
                pass

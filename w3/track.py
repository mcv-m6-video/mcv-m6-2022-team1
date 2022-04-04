import json
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from SORT import Sort
from utils import iou, select_bboxes
# from SORT_models import Sort


class Track:
    def __init__(self, track_id: int, first_bbox: np.ndarray, start_frame: int):
        """
        Initialise a track object. Bboxes must be in the XYXY format.

        Parameters
        ----------
        track_id: int
            Unique track identifier. Uniqueness is not enforced by the class
            itself, but a wrapper object (handle with care!).
        first_bbox: np.ndarray
            First box to consider for the track (in XYXY format).
        """
        self.track_id = track_id
        self.start_frame = start_frame
        self.bboxes = [first_bbox]
        self.centers = [self._get_center(first_bbox)]

    def __str__(self):
        return f"Track #{self.track_id} with {len(self.bboxes)} boxes\n"

    def __repr__(self):
        return str(self)

    @staticmethod
    def _get_center(bbox):
        return bbox.reshape((2, 2)).mean(0)

    def append_bbox(self, new_bbox: np.ndarray):
        self.bboxes.append(new_bbox)
        self.centers.append(self._get_center(new_bbox))

    def get_last_bbox(self):
        return self.bboxes[-1]

    def get_last_center(self):
        return self.centers[-1]

    def get_id(self):
        return self.track_id

    def cvt_mots(self):
        return [[   # FIXME: Confidences should be added sooner or later
            self.start_frame + ii, self.track_id, x[0], x[1], x[2] - x[0],
            x[3] - x[1], 1, -1, -1, -1] for ii, x in enumerate(self.bboxes)
        ]


class MaxOverlapTracker:
    def __init__(
            self,
            start_frame: int,
            end_frame: int,
            thresh: float = 0.5
    ) -> None:
        """
        Given a detection input file in COCO format, this class performs
        maximum overlap tracking of the detected objects.

        Parameters
        ----------
        start_frame: int
            First frame to consider
        end_frame: int
            Final frame to consider
        thresh: float
            IoU acceptance threshold
        """
        self.alive_tracks = []
        self.dead_tracks = []

        self.start_frame = start_frame
        self.end_frame = end_frame

        self.current_track = 0
        self.thresh = thresh

    def _numpy_alive_boxes(self) -> np.ndarray:
        """
        Get a Numpy representation of the set of boxes in tracks that are still
        alive.

        Returns
        -------
        np.ndarray
            Set of "alive" boxes in XYXY format.

        """
        alive = [x.get_last_bbox() for x in self.alive_tracks]
        alive = np.asarray(alive)

        return alive

    def _add_box_to_track(
            self,
            track_index: int,
            new_bbox: np.ndarray
    ) -> None:
        """
        Adds a new bounding box to an existing track.

        Parameters
        ----------
        track_index: int
            Index of the tracking object to append the box to.
        new_bbox: np.ndarray
            Bounding box to be appended in XYXY format.
        """
        self.alive_tracks[track_index].append_bbox(new_bbox)

    def _kill_tracks(self, track_indices: list):
        for ii in track_indices:
            self.dead_tracks.append(self.alive_tracks[ii])
        self.alive_tracks = [x for ii, x in enumerate(self.alive_tracks)
                             if ii not in track_indices]

    def _add_tracks(self, track_list: list):
        for ii in track_list:
            self.alive_tracks.append(ii)

    def _merge_tracks(self) -> list:
        return self.alive_tracks + self.dead_tracks

    def track_objects(self, detections: list):
        mots_style = coco2motsXYXY(detections)
        first = np.asarray(
            mots_style[mots_style["frame"] == self.start_frame]
            [["left", "top", "right", "bot"]]
        )

        self.alive_tracks = [
            Track(ii, bbox, self.start_frame) for ii, bbox in enumerate(first)
        ]

        self.current_track = len(self.alive_tracks)

        for current_frame in range(self.start_frame + 1, self.end_frame + 1):
            current_bboxes = np.asarray(
                mots_style[mots_style["frame"] == current_frame]
                [["left", "top", "right", "bot"]]
            )

            # TODO: NMS <here> could work

            prev_bboxes = self._numpy_alive_boxes()
            iou_boxes = iou(prev_bboxes, current_bboxes)
            chosen_preds = select_bboxes(iou_boxes, self.thresh)

            new_tracks = []

            for pred, jj in enumerate(chosen_preds):
                if jj >= 0:
                    self._add_box_to_track(jj, current_bboxes[pred])
                else:
                    self.current_track += 1
                    new_tracks.append(Track(
                        self.current_track,
                        current_bboxes[pred],
                        current_frame
                    ))

            to_kill = np.where(
                np.isin(np.arange(len(prev_bboxes)), chosen_preds, invert=True)
            )[0].tolist()

            self._kill_tracks(to_kill)
            self._add_tracks(new_tracks)

    def output_tracks(self, out_path: str):
        all_tracks = self._merge_tracks()
        all_mots = []
        for track in all_tracks:
            all_mots += track.cvt_mots()

        pd.DataFrame(all_mots).to_csv(out_path, header=False, index=False)


def read_detections(json_file: str) -> list:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def coco2motsXYXY(
        coco_detections: list,
        conf_purge: float = None
) -> pd.DataFrame:
    rows = [
        [x["image_id"], x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2],
         x["bbox"][1] + x["bbox"][3], x["score"]]
        for x in coco_detections
    ]
    mots = pd.DataFrame(rows, columns=["frame", "left", "top", "right",
                                       "bot", "confidence"])
    mots = mots.sort_values(by="frame")
    return mots


# def track_max_overlap(
#         data,
#         init_frame_id,
#         last_frame_id,
#         IoU_threshold=0.2,
#         score_threshold=0.9
# ):
#     # Assumes first frame as initialization
#
#     tracking_list = list()  # list of Track objects
#     track_id = 0
#
#     for frame_id in tqdm(range(init_frame_id, last_frame_id + 1)):
#         frame_detections = [x for x in data if x["image_id"] == frame_id]
#         frame_detections = [x for x in frame_detections if x["score"] > score_threshold]
#
#         num_detections = len(frame_detections)
#         idx_assigned = [-1] * num_detections
#
#         for ii, object_in_frame in enumerate(frame_detections):
#             if frame_id == init_frame_id:  # initial frame
#                 # appends new track
#                 new_track = Track(track_id)
#                 new_track.append_bbox(object_in_frame["bbox"])
#                 new_track.append_frame_id_appearence(frame_id)
#                 tracking_list.append(new_track)
#                 track_id += 1
#
#             else:
#                 # check IoU of every detected object_in_frame in new frame with previous detected bboxes in
#                 # tracking_list
#                 for track_prev in tracking_list:
#                     iou_between_currentNprev = iou(object_in_frame["bbox"], track_prev.bbox[-1])
#                     # FIXME: use higher IoU instead of just this
#                     if iou_between_currentNprev > IoU_threshold:
#                         if track_prev.frame_id_appearence[-1] == frame_id :  # if updated in this one
#                             idx_assigned[ii] = 0  # this object was assigned
#                         else:
#                             track_prev.append_bbox(object_in_frame["bbox"])  # updates new bbox position
#                             idx_assigned[ii] = 0  # this object was assigned
#                             track_prev.append_frame_id_appearence(frame_id)
#
#         # unassigned new objects -> new track
#         if frame_id != init_frame_id:
#             for jj, is_assigned in enumerate(idx_assigned):
#                 if is_assigned == -1:  # if not assigned
#                     new_track = Track(track_id)
#                     new_track.append_bbox(frame_detections[jj]["bbox"])
#                     new_track.append_frame_id_appearence(frame_id)
#                     tracking_list.append(new_track)
#                     track_id += 1
#
#     return tracking_list


# def iou(gt: list, pred: list):
#     gt_x1, gt_y1, gt_w, gt_h = gt
#     pd_x1, pd_y1, pd_w, pd_h = pred
#
#     gt_x2 = gt_x1 + gt_w
#     pd_x2 = pd_x1 + pd_w
#
#     gt_y2 = gt_y1 + gt_h
#     pd_y2 = pd_y1 + pd_h
#
#     xA = max(gt_x1, pd_x1)
#     yA = max(gt_y1, pd_y1)
#     xB = min(gt_x2, pd_x2)
#     yB = min(gt_y2, pd_y2)
#
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#
#     boxAArea = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
#     boxBArea = (pd_x2 - pd_x1 + 1) * (pd_y2 - pd_y1 + 1)
#
#     return interArea / float(boxAArea + boxBArea - interArea)


def visualize_overlap(track_list, frame_loader, num_of_colors=200):
    color_list = [tuple(np.random.choice(range(256), size=3)) for ic in range(num_of_colors)]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter("overlap.mp4", fourcc, 10, (1920, 1080))

    trail_points = list()
    trail_counter = 0
    for img_frame_id, img in tqdm(frame_loader):
        if img_frame_id == 900:
            print("STOP")
            break
        img = np.array(img)
        bboxes_to_draw = list()

        for track in track_list:
            # if track has stored the frame id, extract bbox and append to draw itvideo.write(img)
            try:
                index = track.frame_id_appearence.index(img_frame_id)
                associated_id = track.id
                bboxes_to_draw.append((track.bbox[index], associated_id))
                trail_points.append((track.bbox[index], associated_id))
            except ValueError:
                pass

        for (x, y, w, h), id_track in bboxes_to_draw:
            cv2.circle(img, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), -1)
            cv2.putText(img, f"id: {id_track}", (int(x + (w / 2)), int(y + (h / 2))), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (int(color_list[id_track % num_of_colors][0]), int(color_list[id_track % num_of_colors][1]),
                         int(color_list[id_track % num_of_colors][2])), 2, cv2.LINE_AA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)

        # for (x, y, w, h), id_track in trail_points:
        #     cv2.circle(img, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # video.write(img)
        #
        # video.write(img)
        # if trail_counter%50 == 0:
        #     trail_counter = 0
        #     trail_points.clear()
        # trail_counter += 1

    cv2.destroyAllWindows()
    video.release()


def visualize_KF(track_list, frame_loader, num_of_colors=200):
    color_list = [tuple(np.random.choice(range(256), size=3)) for ic in range(num_of_colors)]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter("overlap.mp4", fourcc, 10, (1920, 1080))

    trail_points = list()
    trail_counter = 0
    for img_frame_id, img in tqdm(frame_loader):
        if img_frame_id == 900:
            print("STOP")
            break
        img = np.array(img)
        bboxes_to_draw = list()

        for track in track_list:
            # if track has stored the frame id, extract bbox and append to draw itvideo.write(img)
            try:
                index = track[0]
                associated_id = track[1]
                bboxes_to_draw.append((track[2:6], associated_id))
                trail_points.append((track[2:6], associated_id))
            except ValueError:
                pass

        for (x, y, w, h), id_track in bboxes_to_draw:
            cv2.circle(img, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), -1)
            cv2.putText(img, f"id: {id_track}", (int(x + (w / 2)), int(y + (h / 2))), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (int(color_list[id_track % num_of_colors][0]), int(color_list[id_track % num_of_colors][1]),
                         int(color_list[id_track % num_of_colors][2])), 2, cv2.LINE_AA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)

        # for (x, y, w, h), id_track in trail_points:
        #     cv2.circle(img, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # video.write(img)
        #
        # video.write(img)
        # if trail_counter%50 == 0:
        #     trail_counter = 0
        #     trail_points.clear()
        # trail_counter += 1

    cv2.destroyAllWindows()
    video.release()


def eval_file(track_list, init_frame_id, last_frame_id, csv_file):
    for frame_id in tqdm(range(init_frame_id, last_frame_id + 1)):
        for track in track_list:
            try:
                index = track.frame_id_appearence.index(frame_id)
                id_csv = track.id
                bbox = track.bbox[index]
                csv_row = [str(frame_id), str(id_csv), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), "-1",
                           "-1", "-1", "-1"]
                with open(csv_file, 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(csv_row)
            except ValueError:
                pass


def track_KF(data, init_frame_id, last_frame_id, IoU_threshold=0.2, score_threshold =0.9, model_type = 0):
    # Assumes first frame as initialization

    bb_id_updated = []

    # tracker = Sort(model_type = model_type)
    tracker = Sort()
    
    for frame_id in tqdm(range(init_frame_id, last_frame_id + 1)):
        

        frame_detections = [x for x in data if x["image_id"] == frame_id]
        frame_detections = [x for x in frame_detections if x["score"] > score_threshold]

        # dets = np.array([x['bbox'] for x in frame_detections])
        a = np.array([x['bbox'] for x in frame_detections])
        b = np.array([x['score'] for x in frame_detections])
        b = b.reshape((b.shape[0], 1))
        dets = np.append(a, b, axis=1)
        dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2] for the tracker input
        
        trackers = tracker.update(dets)
        
        for bb_dets, bb_update in zip(frame_detections, trackers):
            # bb_id_updated.append([bb_dets['image_id'], bb_dets['category_id'], int(bb_update[4]), bb_update[0], bb_update[1], bb_update[2]-bb_update[0], bb_update[3]-bb_update[1], bb_dets['score']])
            bb_id_updated.append([bb_dets['image_id'], int(bb_update[4])-1, bb_update[0], bb_update[1], bb_update[2]-bb_update[0], bb_update[3]-bb_update[1], -1, -1, -1, -1])
        

    return bb_id_updated



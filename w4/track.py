import json
import csv
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2
from external_lib.SORT import Sort
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

    def __len__(self):
        return len(self.bboxes)

    def __iadd__(self, other):
        self.merge_track(other)

    @staticmethod
    def _get_center(bbox: np.ndarray):
        return bbox.reshape((2, 2)).mean(0)

    def merge_track(self, other):
        self.bboxes += other.bboxes
        self.centers += other.centers

    def append_bbox(self, new_bbox: np.ndarray):
        self.bboxes.append(new_bbox)
        self.centers.append(self._get_center(new_bbox))

    def get_first_bbox(self):
        return self.bboxes[0]

    def get_last_bbox(self):
        return self.bboxes[-1]

    def get_last_center(self):
        return self.centers[-1]

    def get_first_frame(self):
        return self.start_frame

    def get_last_frame(self):
        return self.start_frame + len(self.bboxes)

    def get_id(self):
        return self.track_id

    def get_avg_displ(self):
        centers = np.asarray(self.centers)
        delta = centers[1:] - centers[:1]

        return np.linalg.norm(delta.mean(0), 2)

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
        if len(alive):
            alive = np.asarray(alive)
        else:
            alive = np.zeros((0, 4))

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

        for current_frame in tqdm(range(self.start_frame + 1, self.end_frame + 1),
                                  desc="Tracking progress..."):
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

    def kill_all(self):
        self.dead_tracks += self.alive_tracks
        self.alive_tracks = []

    def output_tracks(self, out_path: str):
        all_tracks = self._merge_tracks()
        all_mots = []
        for track in all_tracks:
            all_mots += track.cvt_mots()

        pd.DataFrame(all_mots).to_csv(out_path, header=False, index=False)

    def cleanup_tracks(self, min_track_length: int, tol: float = 25):
        # Remove any short sequences --> Outliers
        self.dead_tracks = [x for x in self.dead_tracks if len(x) >= min_track_length]

        # Remove static sequences
        self.dead_tracks = [x for x in self.dead_tracks if x.get_avg_displ() > tol]




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

        a = np.array([x['bbox'] for x in frame_detections])
        b = np.array([x['score'] for x in frame_detections])
        b = b.reshape((b.shape[0], 1))
        dets = np.append(a, b, axis=1)
        dets[:, 2:4] += dets[:, 0:2]
        
        trackers = tracker.update(dets)
        
        for bb_dets, bb_update in zip(frame_detections, trackers):
            bb_id_updated.append([
                bb_dets['image_id'], int(bb_update[4])-1,
                bb_update[0], bb_update[1], bb_update[2]-bb_update[0],
                bb_update[3]-bb_update[1], -1, -1, -1, -1
            ])

    return bb_id_updated

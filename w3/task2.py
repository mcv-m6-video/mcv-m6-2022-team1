import json
from tqdm import tqdm
from w1.eval import iou
from track import Track


def read_detections(json_file: str) -> dict():
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def track_max_overlap(data, init_frame_id, last_frame_id, IoU_threshold=0.4):
    # Assumes first frame as initialization

    tracking_list = list()  # list of Track objects
    track_id = 0
    init_flag = 0

    prev_num_detections = 0
    for frame_id in tqdm(range(init_flag, last_frame_id+1)):
        frame_detections = [x for x in data if x["image_id"] == frame_id]
        num_detections = len(frame_detections)

        for object in frame_detections:
            if frame_id == init_frame_id: # initial frame
                # TODO: maybe remove num_detect
                # appends new track
                new_track = Track(track_id, num_detections)
                new_track.append_bbox(object["bbox"])
                tracking_list.append()
                track_id += 1
            else:
                # check IoU of every detected object in new frame with previous detected bboxes in tracking_list
                for track_prev in tracking_list:
                    found_match = 0
                    iou_between_currentNprev = iou(object["bbox"], track_prev.bbox)
                    if iou_between_currentNprev > IoU_threshold:
                        track_prev.append_bbox(object["bbox"]) # updates new bbox position
                        found_match = 1
                if found_match == 0: # if current object cannot be associated to a prev track, it must be a new object
                    new_track = Track(track_id, num_detections)
                    new_track.append(object["bbox"])
                    track_id += 1
















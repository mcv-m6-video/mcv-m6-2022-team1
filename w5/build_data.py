import os
from os import makedirs
from pathlib import Path

from track import read_detections, MaxOverlapTracker
from build_data_utils import create_data_metric_learning, detections_txt2Json

who = 'marcos'

if who == 'dani':
    detections_path = r"E:\Master\M6 - Video analysis\Project/data/" \
                      r"coco_instances_results.json"
    frame_path = Path(r"E:\Master\M6 - Video analysis\Project/AICity_data/"
                      r"train/S03/c010/vdo_frames")
    out_data = Path(r"E:\Master\M6 - Video analysis\Project/data")

    makedirs(out_data, exist_ok=True)
elif who == 'pau':
    detections_path = "/home/pau/Documents/master/M6/project/repo/w3/results/" \
                      "train_holdout_aug/faster_fpn/coco_instances_results.json"
    frame_path = Path("/home/pau/Documents/master/M6/project/data/AICity_data/"
                      "AICity_data/train/S03/c010/vdo_frames")
    out_data = Path("./results/watever")

    makedirs(out_data, exist_ok=True)
else:
    detections_path = "/home/cisu/PycharmProjects/mcv-m6-2022-team1/w5/DATA/train"
    detections_model = "det_mask_rcnn"

sq_list = ["S01", "S03", "S04"]
sq_list = ["S01"]
for sq in sq_list:
    path_to_cam = os.path.join(detections_path, sq)  # ../S01
    subfolders = [f.path for f in os.scandir(path_to_cam) if f.is_dir()]

    for sub in subfolders:  # sub: '/home/cisu/PycharmProjects/mcv-m6-2022-team1/w5/DATA/train/S01/c005'

        detections_txt2Json(os.path.join(sub, "det", detections_model + '.txt'), os.path.join(sub, "det", detections_model + ".json"))
        data = read_detections(os.path.join(sub, "det", detections_model + ".json"))
        tracker = MaxOverlapTracker(data[0]["image_id"], data[-1]["image_id"])
        tracker.track_objects(data)

        tracker.output_tracks("./csvofshame.csv")

        create_data_metric_learning("./csvofshame.csv", os.path.join(sub, "vdo_frames"),
                                    os.path.join(sub, "croped_imgs"))

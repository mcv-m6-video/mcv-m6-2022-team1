from os import makedirs
from pathlib import Path
 
from track import read_detections, visualize_overlap, eval_file, MaxOverlapTracker
from data import FrameLoader
import numpy as np

who = 'pau'

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
    detections_path = "/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data/" \
                      "coco_instances_results.json"
    frame_path = Path("/home/cisu/PycharmProjects/mcv-m6-2022-team1/"
                      "AICity_data/train/S03/c010/vdo_frames")
    out_data = Path("/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data")
    
    makedirs(out_data, exist_ok=True)

data = read_detections(detections_path)

tracker = MaxOverlapTracker(536, 2141)
tracker.track_objects(data)

tracker.output_tracks("./csvofshame.csv")
# tracking = track_max_overlap(data, 536, 2141)
# print(len(tracking))
# np.savez(out_data / "tracking_list.npz", tracking)
# # tracking = np.load(out_data / "tracking_list.npz")
# eval_file(track_list=tracking, init_frame_id=536, last_frame_id=2141, csv_file=out_data / "max_overlap.cvs")
#
# loader = FrameLoader(frame_path, .25, "upper")
# visualize_overlap(tracking, loader)

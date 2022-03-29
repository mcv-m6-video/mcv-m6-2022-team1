from os import makedirs
from pathlib import Path

from track import read_detections, track_max_overlap, visualize_overlap, eval_file
from w2.data import FrameLoader
import numpy as np

detections_path = "/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data/coco_instances_results.json"
frame_path = Path(
    "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/S03/c010/vdo_frames"
)
out_data = Path("/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data")
makedirs(out_data, exist_ok=True)

data = read_detections(detections_path)

tracking = track_max_overlap(data, 536, 836)
print(len(tracking))
np.savez(out_data / "tracking_list.npz", tracking)
# tracking = np.load(out_data / "tracking_list.npz")
# eval_file(track_list=tracking, init_frame_id=536, last_frame_id=2141, csv_file=out_data / "max_overlap.cvs")

loader = FrameLoader(frame_path, .25, "upper")
visualize_overlap(tracking, loader)

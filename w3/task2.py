from os import makedirs
from pathlib import Path

from track import read_detections, track_max_overlap, visualize_overlap
from w2.data import FrameLoader
import numpy as np

frame_path = Path(
    "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/S03/c010/vdo_frames"
)
out_data = Path("/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data")

makedirs(out_data, exist_ok=True)

data = read_detections("/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/prediction_6.json")

tracking = track_max_overlap(data, 536, 2141)

np.savez(out_data / "tracking_list.npz", tracking)
# tracking = np.load(out_data / "tracking_list.npz")
loader = FrameLoader(frame_path, .25, "upper")
visualize_overlap(tracking, loader)

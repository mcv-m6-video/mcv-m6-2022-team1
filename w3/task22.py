from os import makedirs
from pathlib import Path
 
from track import read_detections, track_max_overlap, visualize_overlap, eval_file, track_KF
from data import FrameLoader
import numpy as np
import pandas as pd

who = 'dani'

if who == 'dani':
    detections_path = "E:\Master\M6 - Video analysis\Project/data/coco_instances_results.json"
    frame_path = Path("E:\Master\M6 - Video analysis\Project/AICity_data/train/S03/c010/vdo_frames"    )
    out_data = Path("E:\Master\M6 - Video analysis\Project/data")
    
    makedirs(out_data, exist_ok=True)
else:
    detections_path = "/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data/coco_instances_results.json"
    frame_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/S03/c010/vdo_frames"
    )
    out_data = Path("/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data")
    
    makedirs(out_data, exist_ok=True)

data = read_detections(detections_path)

tracking_vel = track_KF(data, 536, 2141,  model_type = 0)
tracking_acc = track_KF(data, 536, 2141,  model_type = 1)

labels = ['image_id','category_id','id','left','top','width','height','score']
tracking_vel_pd = pd.DataFrame(tracking_vel,columns=labels)
tracking_acc_pd = pd.DataFrame(tracking_acc,columns=labels)

print(len(tracking_vel))
# np.savez(out_data / "tracking_listKF.npz", tracking)
# tracking = np.load(out_data / "tracking_list.npz")
# eval_file(track_list=tracking, init_frame_id=536, last_frame_id=2141, csv_file=out_data / "max_overlap.cvs")

# loader = FrameLoader(frame_path, .25, "upper")
# visualize_overlap(tracking, loader)

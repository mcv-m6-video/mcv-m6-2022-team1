import csv

from os import makedirs
from pathlib import Path
 
from track import read_detections, visualize_overlap, eval_file, track_KF, visualize_KF
from data import FrameLoader
import numpy as np
import pandas as pd

who = 'pau'

if who == 'dani':
    detections_path = r"E:\Master\M6 - Video analysis\Project/data/" \
                      r"coco_instances_results.json"
    frame_path = Path(r"E:\Master\M6 - Video analysis\Project/"
                      r"AICity_data/train/S03/c010/vdo_frames")
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
    detections_path = "/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data/coco_instances_results.json"
    frame_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/S03/c010/vdo_frames"
    )
    out_data = Path("/home/cisu/PycharmProjects/mcv-m6-2022-team1/w3/data")
    
    makedirs(out_data, exist_ok=True)

data = read_detections(detections_path)

tracking_vel = track_KF(data, 536, 2141,  model_type = 0)
# tracking_acc = track_KF(data, 536, 2141,  model_type = 1)

# labels = ['image_id','category_id','id','left','top','width','height','score']
labels = ['image_id','id','left','top','width','height','score','x','y','z']
tracking_vel_pd = pd.DataFrame(tracking_vel,columns=labels)

print(len(tracking_vel))
np.savez(out_data / "tracking_listKF.npz", tracking_vel)

# loader = FrameLoader(frame_path, .25, "upper")
# visualize_KF(tracking_vel, loader)

# open the file in the write mode
f = open(out_data / "tracking_KF.csv", 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
for row in tracking_vel:
    writer.writerow(row)

# close the file
f.close()

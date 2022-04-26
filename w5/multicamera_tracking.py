import cv2
import re
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from track import Track
from utils import data, viz
from datasets import CarIdProcessed
from models import CarIdResnet

#%%

# TODO Load tracks into a data structure
def load_tracks_from_file(track_file: str):
    all_tracks = data.load_annotations(track_file)
    tracks = all_tracks["ID"].unique()

    loaded_tracks = []

    for track_id in tracks:
        split_tracks = all_tracks[all_tracks["ID"] == track_id]
        split_tracks = split_tracks.sort_values(by="frame")
        first_frame = split_tracks["frame"].iloc[0]

        car_frames = split_tracks[["left", "top", "width", "height"]].to_numpy()
        car_frames[:, 2:] = car_frames[:, 0:2] + car_frames[:, 2:]

        current_track = Track(track_id, car_frames[0], first_frame)
        for bbox in car_frames[1:]:
            current_track.append_bbox(bbox)
        loaded_tracks.append(current_track)


# TODO Extract car frames for each track
def extract_track_cars(track_file: str, frame_path: str, out_path: str):

    all_tracks = data.load_annotations(track_file)
    frames = all_tracks["frame"].unique()

    out_path = Path(out_path)
    frame_path = Path(frame_path)

    for track in all_tracks["ID"].unique():
        (out_path / f"{track}").mkdir(exist_ok=True, parents=True)

    for frame in frames:
        img = cv2.imread(str(frame_path / f"{frame:05}.jpg"))
        split_frames = all_tracks[all_tracks["frame"] == frame]

        car_frames = split_frames[["left", "top", "width", "height"]].to_numpy()
        car_frames[:, 2:] = car_frames[:, 0:2] + car_frames[:, 2:]
        car_frames = car_frames.astype(int)

        car_ids = split_frames["ID"].to_numpy()

        for (x1, y1, x2, y2), car_id in zip(car_frames, car_ids):
            car = img[y1:y2, x1:x2]
            cv2.imwrite(str(out_path / f"{car_id}" / f"{frame}.jpg"), car)

#%%

results_path = "/home/pau/Documents/master/M6/project/repo/w4/results/train_models_testS03/faster_fpn"
results_path = Path(results_path)
root_dataset_path = "/home/pau/Documents/datasets/aicity/train"
root_dataset_path = Path(root_dataset_path)

#%%
sequence_re = re.compile("(S\d\d)")
camera_re = re.compile("(c\d\d\d)")

for camera in results_path.glob("ai_citiesS??c???"):
    folder_name = camera.parts[-1]

    camera_name = camera_re.search(folder_name)[0]
    sequence_name = sequence_re.search(folder_name)[0]

    extract_track_cars(
        str(camera / "track_purge.txt"),
        str(root_dataset_path / f"{sequence_name}" / f"{camera_name}" / "vdo_frames"),
        str(camera / "cars")
    )

#%%

# TODO Extract features for each car frame


device = torch.device("cuda")

model_weights = "/home/pau/Documents/master/M6/project/repo/w5/results/margin02/weights/weights_5.pth"
model = CarIdResnet([512, 256])
weights_dict = torch.load(model_weights)
model.load_state_dict(weights_dict)
model = model.to(device)

dataset = CarIdProcessed(str(results_path))
dataloader = DataLoader(
    dataset,
    batch_size=50,
    shuffle=False,
    num_workers=2,
)

features = []
labels = []

model.eval()
with torch.no_grad():
    for img, label in tqdm(dataloader, desc="Progress"):
        img = img.to(device)
        features.append(model(img).detach().cpu().numpy())
        labels.append(label)

features = np.vstack(features)
labels = np.concatenate(labels)

#%%

# TODO Cluster features into a single describing feature vector

unique_labels = np.unique(labels)
average_features = np.empty((len(unique_labels), 256))
std_features = np.empty((len(unique_labels), 256))

for ii, label in enumerate(unique_labels):
    current_ind = labels == label
    current_fts = features[current_ind]

    average_features[ii] = np.mean(current_fts, axis=0)
    std_features[ii] = np.std(current_fts, axis=0)


#%%


# TODO Identify cars

distances = cdist(average_features, average_features, metric="euclidean")
merger_candidates = distances <= 0.3
merger_candidates = np.triu(merger_candidates, k=1)

viz.plot_embedding_space(average_features, unique_labels.astype(int), -1)
viz.plot_heatmap_matrix(distances)

#%%




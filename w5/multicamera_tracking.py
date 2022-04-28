import cv2
import re
import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from scipy.spatial.distance import cdist
from pytorch_metric_learning import testers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from track import Track
from utils import data, viz
from datasets import CarIdProcessed, CarIdLargestFrame
from models import CarIdResnet


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


def compute_features(weights_path: str, dataset):
    device = torch.device("cuda")
    model = CarIdResnet([512, 256])

    weights_dict = torch.load(weights_path)
    model.load_state_dict(weights_dict)
    model = model.to(device)

    tester = testers.BaseTester()
    features, labels = tester.get_all_embeddings(dataset, model)
    features = features.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().flatten()

    return features, labels



def main(args):
    results_path = Path(args.cam_folder)
    root_dataset_path = Path(args.gt_path)

    sequence_re = re.compile("(S\d\d)")
    camera_re = re.compile("(c\d\d\d)")

    # Extract camera frames # # # # # # # # # # # # # # # # # # # # # # # # # #
    print("Extract frames...")
    for camera in results_path.glob("ai_citiesS??c???"):
        folder_name = camera.parts[-1]

        camera_name = camera_re.search(folder_name)[0]
        sequence_name = sequence_re.search(folder_name)[0]

        extract_track_cars(
            str(camera / args.track_filename),
            str(root_dataset_path / f"{sequence_name}" / f"{camera_name}" / "vdo_frames"),
            str(camera / "cars")
        )
    # Compute metric learning features  # # # # # # # # # # # # # # # # # # # #
    print("Compute features...")
    if args.aggregate_mode == "average":
        dataset = CarIdProcessed(str(results_path))
    else:
        dataset = CarIdLargestFrame(str(results_path))

    features, labels = compute_features(args.weights_path, dataset)

    unique_labels = np.unique(labels)
    average_features = np.empty((len(unique_labels), 256))

    if args.aggregate_mode == "average":
        for ii, label in enumerate(unique_labels):
            current_fts = features[labels == label]
            average_features[ii] = np.mean(current_fts, axis=0)


    # Compute distances # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print("Compute distances...")
    distances = cdist(average_features, average_features, metric="euclidean")
    merger_candidates = distances <= 0.3
    merger_candidates = np.triu(merger_candidates, k=1)

    # viz.plot_embedding_space(average_features, unique_labels, -1)
    # viz.plot_heatmap_matrix(distances)

    merger_indices = np.where(merger_candidates)

    equivalence_table = [[x] for x in range(len(unique_labels))]

    for aa, bb in zip(*merger_indices):
        equivalence_table[aa].append(bb)
        equivalence_table[bb].append(aa)

    equivalence_table = [list(set(x)) for x in equivalence_table]

    conversion = {}

    for label, equivalences in enumerate(equivalence_table):
        newlabel = equivalences[0]
        camera, actual_label = dataset.get_camera_and_label(label)
        conversion[(camera, actual_label)] = newlabel

    sequence = sequence_re.search(
        str(list(results_path.glob("ai_citiesS??c???"))[0])
    ).group(1)

    camera_re = re.compile("c(\d\d\d)")

    all_cameras = [int(camera_re.search(str(x)).group(1)) for x in results_path.glob("ai_citiesS??c???")]

    for camera in all_cameras:
        camera_path = results_path / f"ai_cities{sequence}c{camera:03d}"

        anns = data.load_annotations(str(camera_path / "track_purge.txt"))

        id_column = anns.ID.to_numpy()

        for label in anns["ID"].unique():
            id_column[id_column == label] = conversion[(camera, label)]
        anns.ID = id_column

        anns.to_csv(str(camera_path / "track_multicam.txt"), header=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluate a track in MOT format",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gt_path",
        type=str,
        help="Path to the ground truth data. Can either be a MOT track or an"
             "xml file such as the one in the provided data",
    )
    parser.add_argument(
        "weights_path",
        type=str,
        help="Path to the metric learning weights file",
    )
    parser.add_argument(
        "cam_folder",
        type=str,
        help="Path to the prediction file. A MOT track in txt format",
    )
    parser.add_argument(
        "track_filename",
        type=str,
        help="Track filename (to use whatever track is in the folder instead)",
    )
    parser.add_argument(
        "aggregate_mode",
        type=str,
        help="Metric learning aggregation",
    )

    args = parser.parse_args()
    main(args)

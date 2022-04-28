import re

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import List, AnyStr


class CarIdDataset(Dataset):
    TFORMS_TRAIN = transforms.Compose([
        transforms.ColorJitter(brightness=.3, hue=.3),
        transforms.RandomResizedCrop(224, (0.2, 1.0)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    TFORMS_TEST = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, path: str, sequences: List[AnyStr], split: str = "train"):
        root_path = Path(path)

        self.labels = []
        self.paths = []

        for seq in sequences:
            for camera in (root_path / "train" / seq).glob("*"):
                for car in (camera / "cars").glob("*"):
                    for image in car.glob("*"):
                        self.paths.append(str(image))
                        self.labels.append(int(car.parts[-1]))
        self.tforms = self.TFORMS_TRAIN if split == "train" else self.TFORMS_TEST

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        img = self.tforms(img)
        return img, self.labels[item]

    def get_labels(self):
        return self.labels


class CarIdProcessed(Dataset):
    camera_re = re.compile(r"c(\d\d\d)")
    id_re = re.compile(r".*/(\d+)/\d+\.jpg")

    TFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, path: str):
        self.labels = []
        self.paths = []

        current_label = 0

        root_path = Path(path)
        for camera in root_path.glob("ai_citiesS??c???"):
            for track_id in (camera / "cars").glob("*"):
                for frame in track_id.glob("*"):
                    self.labels.append(current_label)
                    self.paths.append(str(frame))

                current_label += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        img = self.TFORMS(img)
        return img, self.labels[item]

    def get_labels(self):
        return self.labels

    def get_camera_and_label(self, label):
        candidates = [ii for ii, gtlabel in enumerate(self.labels) if gtlabel == label]
        paths = [self.paths[cand] for cand in candidates]

        camera = self.camera_re.search(paths[0]).group(1)
        actual_label = self.id_re.search(paths[0]).group(1)

        return int(camera), int(actual_label)


class CarIdLargestFrame(Dataset):
    camera_re = re.compile(r"c(\d\d\d)")
    id_re = re.compile(r".*/(\d+)/\d+\.jpg")

    TFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, path: str):
        self.labels = []
        self.paths = []

        current_label = 0

        root_path = Path(path)
        for camera in root_path.glob("ai_citiesS??c???"):
            for track_id in (camera / "cars").glob("*"):
                max_size = -1
                max_frame = None
                for frame in track_id.glob("*"):
                    area = np.prod(Image.open(str(frame)).size)
                    if area > max_size:
                        max_frame = str(frame)
                        max_size = area

                if max_frame is not None:
                    self.paths.append(max_frame)
                    self.labels.append(current_label)
                    current_label += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        img = self.TFORMS(img)
        return img, self.labels[item]

    def get_labels(self):
        return self.labels

    def get_camera_and_label(self, label):
        candidates = [ii for ii, gtlabel in enumerate(self.labels) if gtlabel == label]
        paths = [self.paths[cand] for cand in candidates]

        camera = self.camera_re.search(paths[0]).group(1)
        actual_label = self.id_re.search(paths[0]).group(1)

        return int(camera), int(actual_label)


from pathlib import Path

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
    TFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, path: str):
        self.labels = []
        self.paths = []

        current_label = 1

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

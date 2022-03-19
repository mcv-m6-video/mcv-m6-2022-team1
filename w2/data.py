from pathlib import Path
from typing import Dict, Tuple
import xml.etree.ElementTree as ET
import cv2
import numpy as np

CocoDict = Dict
Height, Width, Channels = int, int, int


class FrameIterator:
    def __init__(self, dataset):
        self.__ind = 0
        self.__dataset = dataset

    def __next__(self):
        if self.__ind < len(self.__dataset):
            sample = self.__dataset[self.__ind]
            self.__ind += 1
            return sample
        else:
            raise StopIteration


class FrameLoader:
    def __init__(
            self,
            frame_path: Path,
            perc: float,
            half: str
    ) -> None:
        self.frame_path = frame_path
        self.images = [x.parts[-1] for x in frame_path.glob("*.jpg")]
        self.images.sort()

        nimgs = int(len(self.images) * perc)

        if half == "lower":
            self.images = self.images[:nimgs]
        else:
            self.images = self.images[nimgs:]

    def __getitem__(self, item) -> np.ndarray:
        img = cv2.imread(str(self.frame_path / self.images[item]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> FrameIterator:
        return FrameIterator(self)

    def probe(self) -> Tuple[Height, Width, Channels]:
        img = self[0]
        return img.shape


def is_parked(element) -> bool:
    attribute = element.find("attribute")
    if attribute is None:
        return False
    elif attribute.text == "false":
        return False
    return True


def generate_gt_from_xml(in_path: Path, ignore_parked: bool = True, ignore_classes: bool = False) -> CocoDict:
    dataset = ET.parse(str(in_path)).getroot()

    # Build coco-compliant dataset in JSON format
    if ignore_classes:
        labels = {
            "moving": 1,
        }
        last_label = 1
    else:
        labels = {}
        last_label = -1

    frames = set()
    ann_id = 0

    # Create the annotations field
    annotations = []
    for track in dataset.findall("track"):
        if ignore_classes:
            obj_label = 1
        else:
            if track.attrib["label"] not in labels:
                last_label += 1
                labels[track.attrib["label"]] = last_label
            obj_label = labels[track.attrib["label"]]

        for num, box in enumerate(track.findall("box")):
            if ignore_parked and track.attrib["label"] == "car" and is_parked(box):
                continue

            # Keep track of images with annotations
            frame = int(box.attrib["frame"]) + 1
            frames.add(frame)

            # Generate a bounding box
            bbox = [
                float(box.attrib["xtl"]),
                float(box.attrib["ytl"]),
                float(box.attrib["xbr"]) - float(box.attrib["xtl"]),
                float(box.attrib["ybr"]) - float(box.attrib["ytl"]),
            ]

            annotations.append({
                "id": ann_id,
                "image_id": frame,
                "category_id": obj_label,
                "bbox": bbox,
                "segmentation": [],
                "keypoints": [],
                "num_keypoints": 0,
                "score": 1,
                "area": bbox[-2] * bbox[-1],
                "iscrowd": 0
            })
            ann_id += 1

    # Create the images field
    images = []
    for ii in frames:
        images.append({
            "id": ii,
            "license": 1,
            "file_name": f"{ii:05}.jpg",
            "height": 1080,
            "width": 1920,
            "date_captured": None,
        })

    # Create the categories field
    categories = []
    for name, cat_id in labels.items():
        categories.append({
            "id": cat_id,
            "name": name,
            "supercategory": "vehicle",
            "keypoints": [],
            "skeleton": [],
        })
    licenses = {
        "id": 1,
        "name": "Unknown",
        "url": "Unknown",
    }
    info = {
        "year": 2022,
        "version": "0.0",
        "description": "Hopefully I did not screw it up this time",
        "contributor": "Nobody",
        "url": "None",
    }

    coco_dict = {
        "info": info,
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "licenses": licenses
    }
    return coco_dict


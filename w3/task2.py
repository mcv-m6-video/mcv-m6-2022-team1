import json
from tqdm import tqdm

def read_detections(json_file: str) -> dict():
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def track_max_overlap(data, IoU_threshold=0.4, ):
    # Assumes first frame as initialization

    tracking_list = list()  # list of Track objects
    track_id = 0
    for frame in tqdm(data):
        num_detections = len(data["annotations"])
        






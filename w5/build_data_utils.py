import csv
import json
import os.path
from tqdm.auto import tqdm

import cv2
import pandas as pd


def detections_txt2Json(tracks_txt_file, output_file_json, confidence_thresh=0.8):
    json_list = []
    with open(tracks_txt_file) as csvFile:
        csvReader = csv.reader(csvFile)
        for rows in csvReader:
            data = {"image_id": int(rows[0]),
                    "bbox": [float(rows[2]), float(rows[3]), float(rows[4]), float(rows[5])],
                    "category_id": 1,
                    "score": float(rows[6])
                    }
            if float(rows[6]) > confidence_thresh:
                json_list.append(data)

    with open(output_file_json, 'w') as fout:
        json.dump(json_list, fout)


def create_data_metric_learning(tracks_csv, frame_path, out_path, base="00000"):
    # read track csv
    df = pd.read_csv(tracks_csv, sep=",", header=None,
                     names=["frame", "ID", "left", "top", "width", "height", "confidence", "null1", "null2", "null3"])

    num_frames = df["frame"].max()

    # go frame by frame
    for i in tqdm(range(1, num_frames + 1), desc="Cropping images progress..."):
        detects_for_frame = df.loc[df['frame'] == i]

        img = cv2.imread(os.path.join(frame_path, base[:-len(str(i))] + str(i) + ".jpg"))
        assert img is not None, f"{os.path.join(frame_path, base[:-len(str(i))] + str(i) + '.jpg')} not found"
        count = 0

        for indx in detects_for_frame.index:
            (x, y, w, h) = (
                detects_for_frame['left'][indx], detects_for_frame['top'][indx], detects_for_frame['width'][indx],
                detects_for_frame['height'][indx])

            detection_ID = detects_for_frame["ID"][indx]

            store_path = os.path.join(out_path, str(detection_ID))

            os.makedirs(str(store_path), exist_ok=True)  # every id in a different folder

            (x, y, w, h) = (int(x), int(y), int(w), int(h))
            cropped_image = img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(store_path, str(count) + ".jpg"), cropped_image)

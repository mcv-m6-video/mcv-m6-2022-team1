import csv
import json
import os.path

import cv2
import pandas as pd


def detections_txt2Json(tracks_txt_file, output_file_json):
    json_list = []
    with open(tracks_txt_file) as csvFile:
        data = {}
        csvReader = csv.reader(csvFile)
        for rows in csvReader:
            data["image_id"] = rows[0]
            data["bbox"] = [rows[2], rows[3], rows[4], rows[5]]
            data["category_id"] = 1
            data["score"] = rows[6]
            json_list.append(data)

    json_list = json_list.reverse()
    with open(output_file_json, 'w') as fout:
        json.dump(json_list, fout)


def create_data_metric_learning(tracks_csv, frame_path, out_path):
    # read track csv
    df = pd.read_csv(tracks_csv, sep=",", header=None,
                     names=["frame", "ID", "left", "top", "width", "height", "confidence", "null1", "null2", "null3"])

    num_frames = df["frame"].max()

    # go frame by frame
    for i in range(1, num_frames + 1):
        detects_for_frame = df.loc[df['frame'] == i]

        img = cv2.imread(os.path.join(frame_path, "" + str(i) + ".jpg"))

        count = 0

        for indx in detects_for_frame.index:
            (x, y, w, h) = (
                detects_for_frame['left'][indx], detects_for_frame['top'][indx], detects_for_frame['width'][indx],
                detects_for_frame['height'][indx])

            detection_ID = detects_for_frame["ID"][indx]

            store_path = os.path.join(out_path, detection_ID)

            os.makedirs(store_path, exist_ok=True)  # every id in a different folder

            cropped_image = img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(store_path, str(count) + ".jpg"), cropped_image)

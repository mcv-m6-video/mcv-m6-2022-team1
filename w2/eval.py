import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pathlib import Path


who = 'pau'

if who == 'pau':
    out_path = Path(
        "/home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/"
        "S03/c010/w2predictions_t1_redone"
    )
    gt_path = Path(
        "/home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/"
        "S03/c010/gt_coco"
    )
elif who == 'dani':
    out_path = Path(
        "E:/Master/M6 - Video analysis/Project/"
        "AICity_data/train/S03/c010/w2predictions"
    )
    gt_path = Path(
        "E:/Master/M6 - Video analysis/Project/"
        "AICity_data/train/S03/c010/gt_coco"
    )
else:
    out_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/"
        "S03/c010/w2predictions"
    )
    gt_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/"
        "S03/c010/gt_coco"
    )

noisy_frames = [
    [x for x in range(1144, 1170)] +
    [x for x in range(1193, 1216)] +
    [x for x in range(1710, 2142)]
]

non_noisy_frames = [x for x in range(535, 2142) if x not in noisy_frames]

for prediction_file in out_path.glob("*.json"):
    print(str(prediction_file))
    coco = COCO(str(gt_path / "gt_moving_onelabel_test_a.json"))

    with open(prediction_file, 'r') as f_pred:
        prediction = json.load(f_pred)

    img_ids = set()
    max_area = {}

    for x in prediction:
        img_ids.add(x["image_id"])
        if x["image_id"] in max_area.keys():
            max_area[x["image_id"]] = max(
                max_area[x["image_id"]],
                x["bbox"][-1] * x["bbox"][-2],
            )
        else:
            max_area[x["image_id"]] = x["bbox"][-1] * x["bbox"][-2]

    sorted_prediction = []
    for x in prediction:
        sorted_prediction.append({
            "image_id": x["image_id"],
            "category_id": x["category_id"],
            "bbox": x["bbox"],
            "score": (x["bbox"][-1] * x["bbox"][-2]) / max_area[x["image_id"]]
        })

    cocodt = coco.loadRes(sorted_prediction)
    cocoeval = COCOeval(coco, cocodt, 'bbox')
    # cocoeval.params.imgIds = non_noisy_frames

    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()


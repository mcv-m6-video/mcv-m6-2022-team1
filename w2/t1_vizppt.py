import cv2
import json

from data import FrameLoader
from pathlib import Path
from background_estimation import StillBackgroundEstimatorGrayscale, \
    cleanup_mask, get_bboxes

from viz import show_image, draw_bboxes

frame_path = Path(
    "/home/pau/Documents/master/M6/project/data/AICity_data/"
    "AICity_data/train/S03/c010/vdo_frames"
)
estimator_path = Path(
    "/home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/"
    "S03/c010/estimators"
)
out_path = Path(
    "/home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/"
    "S03/c010/w2predictions"
)
gt_path = Path(
    "/home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/"
    "S03/c010/gt_coco"
)

pred_path = Path(
    "/home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/"
    "S03/c010/w2predictions"
)

with open(gt_path / "gt_moving_onelabel.json", 'r') as f_json:
    gt_file = json.load(f_json)

with open(pred_path / "prediction_8.json", 'r') as f_json:
    pd_file = json.load(f_json)

# chosen_img = 1617
# gt_bbox = [x["bbox"] for x in gt_file["annotations"] if x["image_id"] == chosen_img]

train_loader = FrameLoader(frame_path, 1.0, "lower")

# estimator = StillBackgroundEstimatorGrayscale(train_loader, 5.0)
# estimator.load_estimator(estimator_path / "estimator.npz")
# estimator.viz_estimator()
#
# img_id, img = train_loader[chosen_img]
# mask = estimator.predict(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
#
# show_image(mask)
# mask = cleanup_mask(mask, 11)
# show_image(mask)
# bboxes = get_bboxes(mask, 50)
#
# draw_bboxes(img, bboxes, gt_bbox)

for id, img in train_loader:
    gt_bbox = [x["bbox"] for x in gt_file["annotations"] if x["image_id"] == id]
    pd_bbox = [x["bbox"] for x in pd_file if x["image_id"] == id]

    draw_bboxes(img, pd_bbox, gt_bbox, str(out_path / f"{id:05}.jpg"))

# ffmpeg -i ../vdo.avi %05d.jpg
print("ADAPTATIVE MODEL")

import cv2
import json

import numpy as np
from tqdm.auto import tqdm
from data import FrameLoader
from pathlib import Path
from background_estimation import StillBackgroundEstimatorGrayscale, AdaptativeEstimatorGrayscale, cleanup_mask, \
    get_bboxes

from viz import show_image, draw_bboxes
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

who = 'pau'

if who == 'pau':
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
        "S03/c010/w2predictions_t2"
    )
    gt_path = Path(
        "/home/pau/Documents/master/M6/project/data/AICity_data/AICity_data/train/"
        "S03/c010/gt_coco"
    )
elif who == 'dani':
    frame_path = Path(
        "E:/Master/M6 - Video analysis/Project/"
        "AICity_data/train/S03/c010/vdo_frames"
    )
    estimator_path = Path(
        "E:/Master/M6 - Video analysis/Project/"
        "AICity_data/train/S03/c010/estimators"
    )
    out_path = Path(
        "E:/Master/M6 - Video analysis/Project/"
        "AICity_data/train/S03/c010/w2predictions"
    )
    gt_path = Path(
        "E:/Master/M6 - Video analysis/Project/"
        "AICity_data/train/S03/c010/gt_coco"
    )
else:
    frame_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/S03/c010/vdo_frames"
    )
    estimator_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/S03/c010/estimators"
    )
    out_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/"
        "S03/c010/w2predictions"
    )
    gt_path = Path(
        "/home/cisu/PycharmProjects/mcv-m6-2022-team1/AICity_data/train/"
        "S03/c010/gt_coco"
    )

train_loader = FrameLoader(frame_path, .25, "lower")
test_loader = FrameLoader(frame_path, .25, "upper")


# estimator.fit()
# For the sake of speed (and since we decided it was a good idea to fit the
# entire training sequence on memory to generate statistics), we have enabled
# a way to reload the estimated backgrounds.


# interestingly enough, areas with high color happen to have higher std. This is
# probably as a result of images not being 0-centered.

print("Testing...")

# TODO grid search??
tol_value = 3.0
rho_values = [0.01, 0.1, 1.0]
prediction = [[] for _ in range(len(rho_values))]

for jj, rho in enumerate(rho_values):
    estimator = AdaptativeEstimatorGrayscale(train_loader)
    estimator.load_estimator(estimator_path / "estimator.npz")

    estimator.viz_estimator()

    for ii, (img_id, img) in tqdm(enumerate(test_loader), desc="Testing progress..."):

        estimator.set_tol(tol_value)
        estimator.set_rho(rho)
        mask = estimator.predict(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

        # A median filter + closing of size n works well enough. We need a heuristic
        # to join close connected regions if they are significantly small. Our
        # solution atm is to just purge any small bboxes.
        mask = cleanup_mask(mask, 11)
        bboxes = get_bboxes(mask, 50)

        prediction[jj] += [{
            "image_id": img_id,
            "category_id": 1,
            "bbox": list(x),
            "score": 1.0
        } for x in bboxes]

    # draw_bboxes(img, bboxes)
    # show_image(mask)
    # show_image(img)

for jj, tol in enumerate(rho_values):
    with open(out_path / f"prediction_{jj}_adaptative.json", 'w') as f_pred:
        json.dump(prediction[jj], f_pred)

coco = COCO(str(gt_path / "gt_moving_onelabel.json"))


for jj, tol in enumerate(rho_values):
    cocodt = coco.loadRes(prediction[jj])
    cocoeval = COCOeval(coco, cocodt, 'bbox')

    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()

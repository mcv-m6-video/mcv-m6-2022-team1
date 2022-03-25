# ffmpeg -i ../vdo.avi %05d.jpg
print("ADAPTATIVE MODEL")

import cv2
import json

import numpy as np
from tqdm.auto import tqdm
from data import FrameLoader
from pathlib import Path
from background_estimation import StillBackgroundEstimatorGrayscale, AdaptativeEstimatorGrayscale, AdaptativeEstimatorMultiCh, cleanup_mask, \
    get_bboxes

from viz import show_image, draw_bboxes
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

who = 'dani'

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
        "S03/c010/w2predictions"
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


colors = ['RGB','CIE','YUV','HSV']

for color in colors:
    train_loader = FrameLoader(frame_path, .25, "lower",color)
    test_loader = FrameLoader(frame_path, .25, "upper",color)

    estimator = AdaptativeEstimatorMultiCh(train_loader)
    # estimator.fit()
    # estimator.save_estimator(estimator_path / f"estimator_{color}_3.npz")
    
    estimator.load_estimator(estimator_path / f"estimator_{color}_3.npz")

    # interestingly enough, areas with high color happen to have higher std. This is
    # probably as a result of images not being 0-centered.
    estimator.viz_estimator()

    print("Testing...")

    tol_values = [0.5 * x for x in range(1, 11)]
    rho_values = list(np.linspace(0.0, 1.0, num=len(tol_values)))
    prediction = [[] for _ in range(len(tol_values))]

    if color == 'RGB':
        CONVERSION = cv2.COLOR_BGR2RGB
    elif color == 'CIE':
        CONVERSION = cv2.COLOR_BGR2LAB
    elif color == 'YUV':
        CONVERSION = cv2.COLOR_BGR2YCrCb
    elif color == 'HSV':
        CONVERSION = cv2.COLOR_BGR2HSV    
        
    for ii, (img_id, img) in tqdm(enumerate(test_loader), desc="Testing progress..."):
        for jj, (tol, rho) in enumerate(zip(tol_values, rho_values)):
            estimator.set_tol(tol)
            estimator.set_rho(rho)
            mask = estimator.predict(cv2.cvtColor(img, CONVERSION))

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


    coco = COCO(str(gt_path / "gt_moving_onelabel_test.json"))

    for jj, tol in enumerate(tol_values):
        with open(out_path / f"prediction_{color}_3_{jj}_adapt.json", 'w') as f_pred:
            json.dump(prediction[jj], f_pred)

    for jj, tol in enumerate(tol_values):
        cocodt = coco.loadRes(prediction[jj])
        cocoeval = COCOeval(coco, cocodt, 'bbox')

        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

for color in colors[1:]:
    train_loader = FrameLoader(frame_path, .25, "lower",color)
    test_loader = FrameLoader(frame_path, .25, "upper",color)

    estimator = AdaptativeEstimatorMultiCh(train_loader,ch=2)
    # estimator.fit()
    # estimator.save_estimator(estimator_path / f"estimator_{color}_2.npz")
    
    estimator.load_estimator(estimator_path / f"estimator_{color}_2.npz")

    # interestingly enough, areas with high color happen to have higher std. This is
    # probably as a result of images not being 0-centered.
    estimator.viz_estimator()

    print("Testing...")

    tol_values = [0.5 * x for x in range(1, 11)]
    rho_values = list(np.linspace(0.0, 1.0, num=len(tol_values)))
    prediction = [[] for _ in range(len(tol_values))]

    if color == 'RGB':
        CONVERSION = cv2.COLOR_BGR2RGB
    elif color == 'CIE':
        CONVERSION = cv2.COLOR_BGR2LAB
    elif color == 'YUV':
        CONVERSION = cv2.COLOR_BGR2YCrCb
    elif color == 'HSV':
        CONVERSION = cv2.COLOR_BGR2HSV  
        
    for ii, (img_id, img) in tqdm(enumerate(test_loader), desc="Testing progress..."):
        for jj, (tol, rho) in enumerate(zip(tol_values, rho_values)):
            estimator.set_tol(tol)
            estimator.set_rho(rho)
            mask = estimator.predict(cv2.cvtColor(img, CONVERSION))

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

    coco = COCO(str(gt_path / "gt_moving_onelabel_test.json"))

    for jj, tol in enumerate(tol_values):
        with open(out_path / f"prediction_{color}_2_{jj}_adapt.json", 'w') as f_pred:
            json.dump(prediction[jj], f_pred)

    for jj, tol in enumerate(tol_values):
        cocodt = coco.loadRes(prediction[jj])
        cocoeval = COCOeval(coco, cocodt, 'bbox')

        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

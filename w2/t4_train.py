import cv2
import json

from tqdm.auto import tqdm
from data import FrameLoader
from pathlib import Path
from background_estimation import StillBackgroundEstimatorGrayscale, StillBackgroundEstimatorMultiCh, \
    cleanup_mask, get_bboxes

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

    estimator = StillBackgroundEstimatorMultiCh(train_loader)
    estimator.fit()
    estimator.save_estimator(estimator_path / f"estimator_{color}_3.npz")
    
for color in colors[1:]:
    train_loader = FrameLoader(frame_path, .25, "lower",color)
    test_loader = FrameLoader(frame_path, .25, "upper",color)

    estimator = StillBackgroundEstimatorMultiCh(train_loader,ch=2)
    estimator.fit()
    estimator.save_estimator(estimator_path / f"estimator_{color}_2.npz")

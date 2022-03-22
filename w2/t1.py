import cv2
import json

from data import FrameLoader
from pathlib import Path
from background_estimation import StillBackgroundEstimatorGrayscale, \
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
    
print(estimator_path.is_dir())
train_loader = FrameLoader(frame_path, .25, "lower")
test_loader = FrameLoader(frame_path, .25, "upper")

estimator = StillBackgroundEstimatorGrayscale(train_loader)

# estimator.fit()
# For the sake of speed (and since we decided it was a good idea to fit the
# entire training sequence on memory to generate statistics), we have enabled
# a way to reload the estimated backgrounds.
estimator.load_estimator(estimator_path / "estimator.npz")

# interestingly enough, areas with high color happen to have higher std. This is
# probably as a result of images not being 0-centered.
estimator.viz_estimator()

print("Testing...")
prediction = []

for ii, (img_id, img) in enumerate(test_loader):
    mask = estimator.predict(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    # A median filter + closing of size n works well enough. We need a heuristic
    # to join close connected regions if they are significantly small. Our
    # solution atm is to just purge any small bboxes.
    mask = cleanup_mask(mask, 11)
    bboxes = get_bboxes(mask, 50)

    prediction += [{
        "image_id": img_id,
        "category_id": 1,
        "bbox": x,
        "score": 1.0
    } for x in bboxes]

    # draw_bboxes(img, bboxes)
    # show_image(mask)
    # show_image(img)

with open(out_path / "prediction.json", 'w') as f_pred:
    json.dump(prediction, f_pred)

coco = COCO(str(gt_path / "gt_moving_onelabel.json"))
cocodt = coco.loadRes(out_path / "prediction.json")
cocoeval = COCOeval(coco, cocodt, "bbox")

cocoeval.evaluate()
cocoeval.accumulate()
cocoeval.summarize()

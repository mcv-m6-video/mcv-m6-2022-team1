import cv2
import json

from tqdm.auto import tqdm
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
    

# bg_subtractor = cv2.BackgroundSubtractorMOG2(detectShadows=True)
# bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
bg_subtractors = [cv2.BackgroundSubtractorMOG2(detectShadows=True), cv2.BackgroundSubtractorKNN(detectShadows=True),
                  cv2.bgsegm.BackgroundSubtractorLSBP(), cv2.bgsegm.BackgroundSubtractorMOG()]
# cv2.bgsegm.BackgroundSubtractorCNT, cv2.bgsegm.BackgroundSubtractorGMG, cv2.bgsegm.BackgroundSubtractorGSOC

test_loader = FrameLoader(frame_path, .25, "upper")

prediction = [[] for _ in range(len(bg_subtractors))]

for ii, (img_id, img) in tqdm(enumerate(test_loader), desc="Testing progress..."):
    for jj, bg_subtractor in enumerate(bg_subtractors):
        mask = bg_subtractor.apply(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        # mask = estimator.predict(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    
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

coco = COCO(str(gt_path / "gt_moving_onelabel.json"))

for jj, bg_subtractor in enumerate(bg_subtractors):
    with open(out_path / f"prediction_SOTA_{jj}.json", 'w') as f_pred:
        json.dump(prediction[jj], f_pred)

for jj, bg_subtractor in enumerate(bg_subtractors):
    print(bg_subtractor)
    cocodt = coco.loadRes(prediction[jj])
    cocoeval = COCOeval(coco, cocodt, 'bbox')

    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()

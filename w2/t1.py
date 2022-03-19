import cv2

from data import FrameLoader
from pathlib import Path
from background_estimation import StillBackgroundEstimatorGrayscale

from viz import show_image

frame_path = Path(
    "/home/pau/Documents/master/M6/project/data/AICity_data/"
    "AICity_data/train/S03/c010/vdo_frames"
)
estimator_path = Path("/home/pau/Documents/master/M6/project/data/AICity_data/"
                      "AICity_data/train/S03/c010/estimators")
train_loader = FrameLoader(frame_path, .25, "lower")
test_loader = FrameLoader(frame_path, .75, "upper")

estimator = StillBackgroundEstimatorGrayscale(train_loader)
estimator.load_estimator(estimator_path / "estimator.npz")

print("Testing...")
for img in test_loader:
    mask = estimator.predict(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    estimator.save_estimator(estimator_path / "estimator.npz")
    show_image(mask)
    show_image(img)


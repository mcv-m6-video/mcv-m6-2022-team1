import cv2

from data import FrameLoader
from pathlib import Path
from background_estimation import StillBackgroundEstimatorGrayscale, \
    cleanup_mask, get_bboxes

from viz import show_image, draw_bboxes

frame_path = Path(
    "/home/pau/Documents/master/M6/project/data/AICity_data/"
    "AICity_data/train/S03/c010/vdo_frames"
)
estimator_path = Path("/home/pau/Documents/master/M6/project/data/AICity_data/"
                      "AICity_data/train/S03/c010/estimators")
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
for ii, img in enumerate(test_loader):
    mask = estimator.predict(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    # A median filter + closing of size n works well enough. We need a heuristic
    # to join close connected regions if they are significantly small.
    mask = cleanup_mask(mask, 11)
    bboxes = get_bboxes(mask, 50)
    draw_bboxes(img, bboxes)
    show_image(mask)
    show_image(img)
    break

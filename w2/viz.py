import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def show_image(img):
    plt.figure(dpi=150)
    plt.imshow(img)
    plt.show()
    plt.close()


def draw_bboxes(
        frame: np.ndarray,
        pred_bboxes: list,
        gt_bboxes: list = None,
) -> None:
    plt.figure()
    plt.imshow(frame)
    plt.axis("off")

    for box in pred_bboxes:
        plt.gca().add_patch(
            patches.Rectangle(
                (box[0], box[1]),
                box[2], box[3],
                color="r",
                alpha=0.3,
            ))

    if gt_bboxes is not None:
        for box in gt_bboxes:
            plt.gca().add_patch(
                patches.Rectangle(
                    (box[0], box[1]),
                    box[2], box[3],
                    color="g",
                    alpha=0.3,
                ))
    plt.show()
    plt.close()

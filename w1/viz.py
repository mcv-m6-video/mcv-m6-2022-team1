import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_boxes(
        frame: np.ndarray,
        gt_coords: np.ndarray,
        pd_coords: np.ndarray,
        out_path: str = None
) -> None:
    """
    Produces a representation of a single frame with the provided ground truth
    and predicted bounding boxes drawn accordingly. If an out_path is provided
    then the output is not shown as a floating plot, being saved instead into
    the specified file instead.

    Parameters
    ----------
    frame: ArrayLike
        Image in RGB format to draw.
    gt_coords: ArrayLike
        Array of shape Nx4 where N is the number of boxes and each component
        is a xyxy format bounding box (left, top, right, bottom coordinates).
        These represent the ground truth boxes.
    pd_coords: ArrayLike
        Array of shape Nx4 where N is the number of boxes and each component
        is a xyxy format bounding box (left, top, right, bottom coordinates).
        These represent the ground truth boxes.
    out_path: str
        Full filename and path of an output image to save the results.

    Returns
    -------
    None
    """
    plt.figure()
    plt.imshow(frame)
    plt.axis("off")

    for ii in range(gt_coords.shape[0]):
        plt.gca().add_patch(
            patches.Rectangle(
                (gt_coords[ii, 0], gt_coords[ii, 1]),
                gt_coords[ii, 2] - gt_coords[ii, 0],
                gt_coords[ii, 3] - gt_coords[ii, 1],
                color="g",
                alpha=0.3,
                ))
    for ii in range(pd_coords.shape[0]):
        plt.gca().add_patch(
            patches.Rectangle(
                (pd_coords[ii, 0], pd_coords[ii, 1]),
                pd_coords[ii, 2] - pd_coords[ii, 0],
                pd_coords[ii, 3] - pd_coords[ii, 1],
                color="r",
                alpha=0.3,
                ))
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()

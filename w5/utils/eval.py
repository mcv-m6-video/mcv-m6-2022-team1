import numpy as np


def iou(bboxes1, bboxes2):
    """
    Fast IoU implementation from https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    We compared it against our own and decided to use this as it is much more
    memory efficient.
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    xa = np.maximum(x11, np.transpose(x21))
    ya = np.maximum(y11, np.transpose(y21))
    xb = np.minimum(x12, np.transpose(x22))
    yb = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum((xb - xa + 1), 0) * np.maximum((yb - ya + 1), 0)

    boxa_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxb_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = inter_area / (boxa_area + np.transpose(boxb_area) - inter_area)

    return iou


def select_bboxes(inter: np.ndarray, thresh: float) -> np.ndarray:
    """
    From the output of IoU (an NxM matrix where N is the number of ground truth
    samples and M is the number of predictions and the value it contains is the
    Intersection Over Union between the n-th and m-th box), generates the chosen
    predicted bounding boxes assuming they are above a set threshold.

    Parameters
    ----------
    inter: np.ndarray
        NxM matrix where N is the number of ground truth
        samples and M is the number of predictions and the value it contains is the
        intersection-over-union between the n-th and m-th box.
    thresh: float
        Acceptance value for Intersection over Union.

    Returns
    -------
    np.ndarray
        M-length array with the selected GT box at each prediction. If a
        prediction box has no corresponding gt, -1 is returned accordingly.
    """
    n, m = inter.shape
    ind_max = np.full(m, -1)

    for ii in range(m):
        above = inter[:, ii] >= thresh
        if np.any(above):
            ind_max[ii] = np.argmax(inter[:, ii])
            inter[ind_max[ii],:] = 0.0

    return ind_max
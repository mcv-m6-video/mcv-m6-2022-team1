import pandas as pd
import numpy as np


def load_annotations(path: str) -> pd.DataFrame:
    """
    Loads a csv-like annotation file with fields ["frame", "ID", "left", "top",
    "width", "height", "confidence", "null1", "null2", "null3"] into a pandas
    dataframe. Check Nvidia AICity challenge readme for further detail.

    Parameters
    ----------
    path: str
        Path string for the input file.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe which contains the loaded csv with only the needed
        columns ["frame", "ID", "left", "top", "width", "height", "confidence"].
    """
    ann = pd.read_csv(
        path,
        sep=",",
        names=["frame", "ID", "left", "top", "width", "height", "confidence", "null1",
               "null2", "null3"]
    )
    ann = ann[["frame", "ID", "left", "top", "width", "height", "confidence"]]
    return ann


def vectorise_annotations(df: pd.DataFrame) -> np.ndarray:
    """
    From a pandas dataframe with the bounding boxes of a single class and a
    single frame, produce a confidence-ordered array of boxes in xyxy format.
    The input dataframe should have the same row indices as load_annotations.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe with the same format as produced by the function
        load_annotations.

    Returns
    -------
    ArrayLike
        Array of shape Nx4 where N is the number of boxes and each component
        is a xyxy format bounding box (left, top, right, bottom coordinates).

    See Also
    --------
    load_annotations : load_annotations function.
    """
    df.sort_values("confidence")
    xyxy_format = (
        df["left"],
        df["top"],
        df["left"] + df["width"],
        df["top"] + df["height"],
    )
    return np.asarray(xyxy_format).T


def iou(
        gt: np.ndarray,
        pred: np.ndarray
) -> np.ndarray:
    """
    Returns the Intersection over Union of a given set of "pred" grid-aligned
    rectangles against a set of reference "gt" grid-aligned rectangles.

    Parameters
    ----------
    gt : ArrayLike
        Set of grid-aligned rectangles to compare against. Provided as a Nx4
        matrix of N points of (x1, y1, x2, y2) coordinates.
    pred : ArrayLike
        Set of grid-aligned rectangles to be compared. Provided as a Mx4
        matrix of M points of (x1, y1, x2, y2) coordinates.

    Returns
    -------
    ArrayLike:
        A NxM matrix with the IoU of each cmp rectangle against each reference
        rectangle.
    """
    n, m = gt.shape[0], pred.shape[0]
    s_gt = np.stack([gt] * m, axis=1)
    s_pred = np.stack([pred] * n, axis=0)

    # Intersection
    intr_x = np.min(np.stack((s_gt[:, :, 2], s_pred[:, :, 2]), axis=0), axis=0) - \
        np.max(np.stack((s_gt[:, :, 0], s_pred[:, :, 0]), axis=0), axis=0)
    intr_x = np.maximum(intr_x, 0)

    intr_y = np.min(np.stack((s_gt[:, :, 3], s_pred[:, :, 3]), axis=0), axis=0) - \
        np.max(np.stack((s_gt[:, :, 1], s_pred[:, :, 1]), axis=0), axis=0)
    intr_y = np.maximum(intr_y, 0)

    intr_t = intr_x * intr_y

    # Union
    area_r = (s_gt[:, :, 2] - s_gt[:, :, 0]) * (s_gt[:, :, 3] - s_gt[:, :, 1])
    area_c = (s_pred[:, :, 2] - s_pred[:, :, 0]) * (s_pred[:, :, 3] - s_pred[:, :, 1])

    union = area_r + area_c - intr_t

    return intr_t / union


def average_precision_frame(
        gt: np.ndarray,
        pred: np.ndarray,
        thresh: float
) -> float:
    """
    Computes the average precision for a single frame and class. The input is
    assumed to be filtered accordingly.

    Parameters
    ----------
    gt: ArrayLike
        Array of shape Nx4 where N is the number of boxes and each component
        is a xyxy format bounding box (left, top, right, bottom coordinates).
        These represent the ground truth boxes.
    pred: ArrayLike
        Array of shape Nx4 where N is the number of boxes and each component
        is a xyxy format bounding box (left, top, right, bottom coordinates).
        These represent a predictions' boxes.
    thresh: float
        Intersection-over-union threshold. Whenever the area is below this
        threshold, the prediction is ignored.

    Returns
    -------
    float
        Average precision for the given bounding boxes.
    """
    inter = iou(gt, pred)
    ind_max = np.where(inter >= thresh, np.argmax(inter, axis=0), -1)

    tp_evol = np.cumsum(ind_max >= 0)
    pre = tp_evol / pred.shape[0]
    rec = tp_evol / gt.shape[0]

    curr_max = -1
    out_pre = np.zeros_like(pre)

    for ii in range(len(pre)):
        curr_max = max(pre[len(pre) - ii - 1], curr_max)
        out_pre[len(pre) - ii - 1] = curr_max

    sampling_points = np.arange(0.0, 1.01, 0.1)
    pre_ind = rec[None, :] >= sampling_points[:, None]
    pre_ind = np.argmax(pre_ind, axis=1)

    ap = sum(out_pre[pre_ind]) / 11

    return ap


def compute_avg_precision(
        gt_path: str,
        pd_path: str,
        alter_prediction: callable = None,
        add_params: dict = None,
) -> dict:
    """
    Computes the average precision from a prediction file w.r.t. a ground truth
    file. Both cases should respect the csv-like annotation format from Nvidia
    AICity challenge.

    Parameters
    ----------
    gt_path: str
        Path to the ground truth file.
    pd_path: str
        Path to the prediction file.
    alter_prediction: callable
        Function to modify the prediction (to test stochastic modifications
        for instance).
    add_params: dict
        Extra parameters for the alter_prediction function.

    Returns
    -------
    dict
        A dictionary containing each frame as key and the computed average
        precision as value.

    See Also
    --------
    load_annotations : load_annotations function.
    """
    truth = load_annotations(gt_path)
    pred = load_annotations(pd_path)

    if alter_prediction is not None:
        pred = alter_prediction(pred, **add_params)

    frame_indices = np.unique(np.concatenate([
        pd.unique(truth["frame"]), pd.unique(pred["frame"])
    ]))

    output = {}

    for frame_id in frame_indices:
        gt_frame = vectorise_annotations(truth[truth["frame"] == frame_id])
        pd_frame = vectorise_annotations(pred[pred["frame"] == frame_id])

        if gt_frame.shape[0] == 0 or pd_frame.shape[0] == 0:
            output[frame_id] = 0.0
        else:
            output[frame_id] = average_precision_frame(gt_frame, pd_frame, 0.5)

    return output


def dropout_predictions(pred: pd.DataFrame, prob: float) -> pd.DataFrame:
    decision = np.random.rand(len(pred)) > prob
    pred = pred[decision]
    return pred

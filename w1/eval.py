import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Should probably refactor to avoid these
import matplotlib.image as mpimg
from viz import draw_boxes


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
        names=["frame", "ID", "left", "top", "width", "height", "confidence",
               "null1", "null2", "null3"]
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
    n, m = inter.shape
    ind_max = select_bboxes(inter, thresh)

    tp_evol = np.cumsum(ind_max >= 0)
    pre = tp_evol / np.arange(1, m + 1)
    rec = tp_evol / gt.shape[0]

    curr_max = -1
    out_pre = np.zeros_like(pre)

    for ii in range(len(pre)):
        curr_max = max(pre[len(pre) - ii - 1], curr_max)
        out_pre[len(pre) - ii - 1] = curr_max

    sampling_points = np.arange(0.0, 1.01, 0.1)
    pre_ind = rec[None,:] >= sampling_points[:, None]
    pre_ind = np.where(np.any(pre_ind, axis=1), np.argmax(pre_ind, axis=1), -1)

    ap = sum(np.where(pre_ind >= 0, out_pre[pre_ind], 0.0)) / 11

    return ap


def compute_avg_precision(
        gt_path: str,
        pd_path: str,
        iou_thresh: float,
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
    iou_thresh: float
        IoU threshold above which a prediction is considered correct.
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

    frame_indices = pd.unique(truth["frame"])
    output = {}

    for frame_id in frame_indices:
        gt_frame = vectorise_annotations(truth[truth["frame"] == frame_id])
        pd_frame = vectorise_annotations(pred[pred["frame"] == frame_id])

        if pd_frame.shape[0] == 0:
            output[frame_id] = 0.0
        else:
            output[frame_id] = average_precision_frame(gt_frame, pd_frame, iou_thresh)

    return output


def dropout_predictions(pred: pd.DataFrame, prob: float) -> pd.DataFrame:
    """
    Removes rows in a pandas DataFrame with probability `prob`.

    Parameters
    ----------
    pred: pd.DataFrame
        Input sample dataset.
    prob: float
        Scalar in range [0, 1.0] that denotes the probability of removing a
        single row.

    Returns
    -------
    pd.DataFrame
        Modified sample dataset with roughly ```prob * len(pred)``` samples.
    """
    decision = np.random.rand(len(pred)) > prob
    pred = pred[decision]
    return pred


def offset_predictions(pred: pd.DataFrame, offset: int) -> pd.DataFrame:
    """
    Offsets all boxes by a fixed integer number.

    Parameters
    ----------
    pred: pd.DataFrame
        Input sample dataset.
    offset: int
        Amount by which to offset each bounding box.

    Returns
    -------
    pd.DataFrame
        Modified sample dataset.

    """
    pred[["left", "top"]] = pred[["left", "top"]] + offset
    return pred


def iou_offset(gt: np.ndarray, offset: int) -> np.ndarray:
    """
    Computes the IoU of a dataset against itself considering a fixed offset.
    This is a simple closed-form solution that allows easy debugging for IoU
    functions.

    Parameters
    ----------
    gt: np.ndarray
        Set of bounding boxes in xyxy format.
    offset: int
        Amount by which to offset the boxes.

    Returns
    -------
    np.ndarray
        Intersection over union value of all the boxes against themselves.
    """
    width = gt[:, 2] - gt[:, 0]
    height = gt[:, 3] - gt[:, 1]

    intersect = (width - offset) * (height - offset)
    union = (2 * width * height) - intersect

    return intersect / union


def test_iou(
        gt_path: str,
        offset: int,
        img_path: str
) -> float:
    """
    Test function to debug IoU.

    Parameters
    ----------
    gt_path: str
        Path to the ground truth data.
    offset: int
        Amount by which to offset bounding boxes for testing.
    img_path: str
        Path to find input images for debugging.

    Returns
    -------
    float
        Proportion of correct IoU values.
    """

    truth = load_annotations(gt_path)

    frame_indices = pd.unique(truth["frame"])
    correct = 0
    total = 0

    for frame_id in frame_indices:
        gt_frame = vectorise_annotations(truth[truth["frame"] == frame_id])
        iou_normal = np.max(iou(gt_frame, gt_frame + offset), axis=1)
        iou_theory = iou_offset(gt_frame, offset)

        if np.any(np.not_equal(iou_normal, iou_theory)):
            print(frame_id, iou_theory, iou_normal)
            img = mpimg.imread(img_path + f"{frame_id:05}.jpg")
            draw_boxes(img, gt_frame, gt_frame + offset)

        correct += np.count_nonzero(iou_normal == iou_theory)
        total += np.prod(iou_normal.shape)

    return correct / total


def test_iou_std(
        gt_path: str,
) -> float:
    """
    Test function to analyze IoU when adding normal noise.

    Parameters
    ----------
    gt_path: str
        Path to the ground truth data.

    Returns
    -------
    float
        IoU mean value.
    """

    truth = load_annotations(gt_path)
    truth = vectorise_annotations(truth)
    correct = 0
    total = 0

    vec_std = np.arange(0,250.1,1)
    mIoU = np.zeros(vec_std.shape)
    area_r = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
    for id, std in enumerate(vec_std):
        ious = []
        print(std)
        for jjj in np.arange(1000):
            # iou_normal = iou(truth, truth + np.random.normal(0,std,truth.shape))
            pred = truth + np.random.normal(0,std,truth.shape)
            
            intr_x = np.min((truth[:, 2], pred[:, 2]), axis=0) - \
                np.max((truth[:, 0], pred[:, 0]), axis=0)
            intr_x = np.maximum(intr_x,0)
            intr_y = np.min((truth[:, 3], pred[:, 3]), axis=0) - \
                np.max((truth[:, 1], pred[:, 1]), axis=0)
            intr_y = np.maximum(intr_y, 0)

            intr_t = intr_x * intr_y

            # Union
            
            area_c = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
            area_c = np.maximum(area_c, 0)
            
            union = area_r + area_c - intr_t

            iou_normal = intr_t / union
        
            # for i in np.arange(iou_normal.shape[0]):
            ious.append(iou_normal)
                    
        mIoU[id] = np.mean(ious)

    plt.plot(vec_std,mIoU)
    plt.title('AP vs normal noise')
    plt.ylabel('Average precision')
    plt.xlabel('Standard deviation [px]')
    
    return mIoU


from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage.filters import convolve as filter2

#
HSKERN = np.array([[1/12, 1/6, 1/12],
                   [1/6,    0, 1/6],
                   [1/12, 1/6, 1/12]], float)

kernelX = np.array([[-1, 1],
                    [-1, 1]]) * .25  # kernel for computing d/dx

kernelY = np.array([[-1, -1],
                    [1, 1]]) * .25  # kernel for computing d/dy

kernelT = np.ones((2, 2))*.25

# def evaluate_flow(flow_noc, flow):
#
#     err = np.sqrt(np.sum((flow_noc[..., :2] - flow) ** 2, axis=2))
#     noc = flow_noc[..., 2].astype(bool)
#     msen = np.mean(err[noc] ** 2)
#     pepn = np.sum(err[noc] > 3) / err[noc].size
#     return msen, pepn

def evaluate_flow(flow, gt_flow):

    u_d = gt_flow[:, :, 0] - flow[:, :, 0]
    v_d = gt_flow[:, :, 1] - flow[:, :, 1]
    error = np.sqrt(u_d ** 2 + v_d ** 2)
    error_non_ocluded = error[gt_flow[:, :, 2] != 0]

    msen = np.mean(error_non_ocluded)

    nu_wrong = np.sum(error_non_ocluded > 3)
    num_pixels_noc = len(error_non_ocluded)

    pepn = (nu_wrong/num_pixels_noc) * 100
    return msen, pepn


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow, method, scale=4):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    if method != 'pyflow':
        # scales if not pyflow xq no se ve una kk
        hsv[..., 2] = np.minimum(mag * scale, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr




def HornSchunck(im1: np.ndarray, im2: np.ndarray, *,
                alpha: float = 0.001, Niter: int = 8,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    im1: numpy.ndarray
        image at t=0
    im2: numpy.ndarray
        image at t=1
    alpha: float
        regularization constant
    Niter: int
        number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    if verbose:
        from .plots import plotderiv
        plotderiv(fx, fy, ft)

#    print(fx[100,100],fy[100,100],ft[100,100])

        # Iteration to reduce error
    for _ in range(Niter):
        # %% Compute local averages of the flow vectors
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
#  common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
# iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)

    # ft = im2 - im1
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft
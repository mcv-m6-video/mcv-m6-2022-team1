import cv2
import numpy as np
import argparse

from pyflow import pyflow
# git clone https://github.com/pathak22/pyflow.git
#  python setup.py build_ext -i
from flowlib import read_flow
from of_utils import evaluate_flow, draw_flow, draw_hsv


def opt_flow(prev_path, post_path, gt_path, method='pyflow'):
    prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    post = cv2.imread(post_path, cv2.IMREAD_GRAYSCALE)
    gt_flow = read_flow(gt_path)

    if method == 'pyflow':
        # needs float type
        img1 = np.atleast_3d(prev.astype(float) / 255.)
        img2 = np.atleast_3d(post.astype(float) / 255.)
        u, v, im2W = pyflow.coarse2fine_flow(img1, img2, 0.012, 0.75, 20, 7, 1, 30, 1)
        opflow = np.dstack((u, v))

    elif method == 'fb':
        opflow = cv2.calcOpticalFlowFarneback(prev, post, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    elif method == 'lk':

        p0 = np.array([[x, y] for y in range(prev.shape[0]) for x in range(prev.shape[1])], dtype=np.float32).reshape((-1, 1, 2))
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, post, p0, None, **lk_params)

        p0 = p0.reshape((prev.shape[0], prev.shape[1], 2))
        p1 = p1.reshape((prev.shape[0], prev.shape[1], 2))
        st = st.reshape((prev.shape[0], prev.shape[1]))

        opflow = p1 - p0
        opflow[st == 0] = 0
    else:
        raise ValueError(f'specify an algorithm, "{method}" is not recognised')

    msen, pepn = evaluate_flow(opflow,gt_flow)
    print(f'MSEN: {msen:.4f}')
    print(f'PEPN: {pepn:.4f} %')

    cv2.imshow(f'flow_{method}', draw_flow(prev, opflow))
    cv2.imshow(f'hsv_{method}', draw_hsv(opflow, method))
    cv2.waitKey(0)


if __name__ == '__main__':
    prev = 'colored_0/000045_10.png'
    post = 'colored_0/000045_11.png'
    gt = '/home/cisu/PycharmProjects/mcv-m6-2022-team1/w4/flow_noc/000045_10.png'

    opt_flow(prev, post, gt, 'fb')

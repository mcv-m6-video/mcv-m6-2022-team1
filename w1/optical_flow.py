import cv2
import numpy as np
import glob
import os
from OpticalFlowToolkit.lib import flowlib as fl


def optical_flow(frames_folder):
    onlyfiles = sorted(filter(os.path.isfile,
                                  glob.glob(frames_folder + '/**/*', recursive=True)))


    feature_params = dict(maxCorners=200,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    for i in range(len(onlyfiles) - 1):

        print(onlyfiles[i],onlyfiles[i+1])

        # Take first frame and find corners in it
        old_frame = cv2.imread(onlyfiles[i])
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        mask = np.zeros_like(old_frame)

        frame = cv2.imread(onlyfiles[i + 1])
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a,b,c,d = int(a), int(b), int(c), int(d)
            mask = cv2.arrowedLine(frame, (a, b), (c, d), (0,255,0), 2)

        cv2.imshow('frame', mask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points

    cv2.destroyAllWindows()


def mse(predicted, gt):
    mse = 0
    N = prev.shape[0]
    M = prev.shape[1]
    mse = np.sum((prev - curr) ** 2) / (M * N)



if __name__ == '__main__':

    # optical_flow('/home/cisu/PycharmProjects/mcv-m6-2022-team1/w1/data_scene_flow/testing/image_2')

    # pred = 'home/cisu/PycharmProjects/mcv-m6-2022-team1/w1/results_opticalflow_kitti/results/LKflow_000045_10.png'
    # gt_ = '/home/cisu/PycharmProjects/mcv-m6-2022-team1/w1/data_scene_flow/training/flow_noc/000045_10.png'
    # fl.visualize_flow(fl.read_flow(gt_), 'RGB')

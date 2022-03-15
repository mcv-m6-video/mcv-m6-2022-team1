import cv2
import numpy as np
import glob
import os
from OpticalFlowToolkit.lib import flowlib as fl
from matplotlib import pyplot as plt


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

        print(onlyfiles[i], onlyfiles[i + 1])

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
            a, b, c, d = int(a), int(b), int(c), int(d)
            mask = cv2.arrowedLine(frame, (a, b), (c, d), (0, 255, 0), 2)

        cv2.imshow('frame', mask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points

    cv2.destroyAllWindows()


def mse(predicted, gt):
    if predicted.shape != gt.shape:
        predicted = np.resize(predicted, gt.shape)

    maks_from_gt = gt[:, :, 2] == 1

    error = np.sqrt(np.power(gt[:, :, :2] - predicted[:, :, :2], 2)).sum(-1)[maks_from_gt]

    return np.mean(error)


def pepn(predicted, gt, threshold=3):
    if predicted.shape != gt.shape:
        predicted = np.resize(predicted, gt.shape)

    maks_from_gt = gt[:, :, 2] == 1

    total = len((gt[:, :, :2] - predicted[:, :, :2])[maks_from_gt])
    error_count = np.sum(((gt[:, :, :2] - predicted[:, :, :2])[maks_from_gt] > threshold))

    return error_count / total


def plot_error(predicted, gt):

    if predicted.shape != gt.shape:
        predicted = np.resize(predicted, gt.shape)

    maks_from_gt = gt[:, :, 2] == 1

    error = np.sqrt(np.power(gt[:, :, :2] - predicted[:, :, :2], 2)).sum(-1)[maks_from_gt]

    for i in range(3):
        plt.figure(i)
        plt.hist(error, bins=30 + i * 10, density=True)
        plt.title('Density of Optical Flow Error')
        plt.xlabel('Optical Flow error')
        plt.ylabel('The Percentage of Pixels')
        plt.savefig(f'./errors_{i}.png')
        plt.show()


if __name__ == '__main__':
    # optical_flow('/home/cisu/PycharmProjects/mcv-m6-2022-team1/w1/data_scene_flow/testing/image_2')

    pred = '../data_of/LKflow_000045_10.png'
    gt_ = '../data_of/000045_10.png'
    gt_flow = fl.read_flow(gt_)
    pred_flow = fl.read_flow(pred)
    print(mse(predicted=pred_flow, gt=gt_flow))
    print(pepn(predicted=pred_flow, gt=gt_flow))
    plot_error(pred_flow, gt_flow)

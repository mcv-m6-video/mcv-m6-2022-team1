from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

from data import FrameLoader

Bbox = Tuple[int, int, int, int]


class StillBackgroundEstimatorGrayscale:
    def __init__(self, loader: FrameLoader, tol: float = 2.5) -> None:
        self.loader = loader
        self.tol = tol
        self.imsize = self.loader.probe()

        self.mean = np.zeros(self.imsize[:2])
        self.variance = np.zeros(self.imsize[:2])

    def __str__(self):
        return f"Still Background estimator with {len(self.loader)} images"

    def fit(self) -> None:
        all_img = np.empty(
            (self.imsize[0] * self.imsize[1], len(self.loader)),
            dtype=float
        )
        for ii, (img_id, img) in tqdm(enumerate(self.loader), desc="Fit progress"):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(float).flatten()
            all_img[:, ii] = img

        self.mean = all_img.mean(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        self.variance = all_img.var(axis=-1).reshape((self.imsize[0], self.imsize[1]))

    def predict(self, img: np.ndarray):
        img = img.astype(float)
        mask = np.abs(img - self.mean) > (self.tol * (np.sqrt(self.variance) + 2))

        return mask

    def save_estimator(self, out_path: Path) -> None:
        np.savez(
            str(out_path),
            mean=self.mean,
            variance=self.variance
        )

    def set_tol(self, tol: float) -> None:
        self.tol = tol

    def load_estimator(self, in_path: Path) -> None:
        npz_arr = np.load(str(in_path))
        self.mean = npz_arr["mean"]
        self.variance = npz_arr["variance"]

    def viz_estimator(self):
        ax = plt.subplot()
        plt.title("Background Estimator Standard Deviation")
        im = ax.imshow(np.sqrt(self.variance), cmap="hot", interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.figure(dpi=150)

        plt.show()
        plt.close()


class StillBackgroundEstimatorMultiCh:
    def __init__(self, loader: FrameLoader, tol: float = 2.5, ch: int = 3) -> None:
        self.loader = loader
        self.tol = tol
        self.imsize = self.loader.probe()

        self.mean = np.zeros((self.imsize[0],self.imsize[1],ch))
        self.variance = np.zeros((self.imsize[0],self.imsize[1],ch))
        self.ch = ch
        
        if ch == 2:
            if loader.color == 'CIE':
                self.startCh = 1
                self.endCh = 3
            elif loader.color == 'YUV':
                self.startCh = 1
                self.endCh = 3
            elif loader.color == 'HSV':
                self.startCh = 0
                self.endCh = 2
        else:
            self.startCh = 0
            self.endCh = 3
            

    def __str__(self):
        return f"Still Background estimator with {len(self.loader)} images"

    def fit(self) -> None:
        all_img = np.empty(
            (self.imsize[0], self.imsize[1], self.ch, len(self.loader)),
            dtype=float
        )
        for ii, (img_id, img) in tqdm(enumerate(self.loader), desc="Fit progress"):
            img = np.array(img)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img = img.astype(float).flatten()
            all_img[:,:,:, ii] = img[:,:,self.startCh:self.endCh]

        # self.mean = all_img.mean(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        # self.variance = all_img.var(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        self.mean = all_img.mean(axis=3)
        self.variance = all_img.var(axis=3)

    def predict(self, img: np.ndarray):
        img = img.astype(float)
        maskCh = np.abs(img - self.mean) > (self.tol * (np.sqrt(self.variance) + 2))
        mask = np.ones((self.imsize[0],self.imsize[1]))
        # mask = np.zeros((self.imsize[0],self.imsize[1]))
        
        for i in range(self.ch):
           mask *= maskCh[:,:,i]
           # mask += maskCh[:,:,i]
            
        return mask

    def save_estimator(self, out_path: Path) -> None:
        np.savez(
            str(out_path),
            mean=self.mean,
            variance=self.variance
        )

    def set_tol(self, tol: float) -> None:
        self.tol = tol

    def load_estimator(self, in_path: Path) -> None:
        npz_arr = np.load(str(in_path))
        self.mean = npz_arr["mean"]
        self.variance = npz_arr["variance"]

    def viz_estimator(self):
        ax = plt.subplot()
        plt.title("Background Estimator Standard Deviation")
        im = ax.imshow(np.sqrt(self.variance), cmap="hot", interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.figure(dpi=150)

        plt.show()
        plt.close()
        
        
class AdaptativeEstimatorGrayscale:
    def __init__(self, loader: FrameLoader, tol: float = 2.5, rho: float = 0.5) -> None:
        self.loader = loader
        self.tol = tol
        self.imsize = self.loader.probe()

        self.mean = np.zeros(self.imsize[:2])
        self.variance = np.zeros(self.imsize[:2])
        self.rho = rho

    def __str__(self):
        return f"Still Background estimator with {len(self.loader)} images"

    def fit(self) -> None:
        all_img = np.empty(
            (self.imsize[0] * self.imsize[1], len(self.loader)),
            dtype=float
        )
        for ii, img in tqdm(enumerate(self.loader), desc="Fit progress"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(float).flatten()
            all_img[:, ii] = img

        self.mean = all_img.mean(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        self.variance = all_img.var(axis=-1).reshape((self.imsize[0], self.imsize[1]))

    def predict(self, img: np.ndarray):
        img = img.astype(float)
        mask = np.abs(img - self.mean) > (self.tol * (np.sqrt(self.variance) + 2))

        # FIXME: maybe not the fastest way
        # calculate new mean and variance for all pixels
        new_mean = self.rho * img + (1 - self.rho) * self.mean
        new_variance = self.rho * (img - self.mean) ** 2 + (1 - self.rho) * self.variance

        # replace only if it is background
        self.mean = np.where(mask == False, new_mean, self.mean)
        self.variance = np.where(mask == False, new_variance, self.variance)

        return mask

    def save_estimator(self, out_path: Path) -> None:
        np.savez(
            str(out_path),
            mean=self.mean,
            variance=self.variance
        )

    def set_tol(self, tol: float) -> None:
        self.tol = tol

    def set_rho(self, rho: float) -> None:
        self.rho = rho

    def load_estimator(self, in_path: Path) -> None:
        npz_arr = np.load(str(in_path))
        self.mean = npz_arr["mean"]
        self.variance = npz_arr["variance"]

    def viz_estimator(self):
        ax = plt.subplot()
        plt.title("Background Estimator Standard Deviation")
        im = ax.imshow(np.sqrt(self.variance), cmap="hot", interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.figure(dpi=150)

        plt.show()
        plt.close()


class AdaptativeEstimatorMultiCh:
    def __init__(self, loader: FrameLoader, tol: float = 2.5, rho: float = 0.5, ch: int = 3) -> None:
        self.loader = loader
        self.tol = tol
        self.imsize = self.loader.probe()

        self.rho = rho
        
        self.mean = np.zeros((self.imsize[0],self.imsize[1],ch))
        self.variance = np.zeros((self.imsize[0],self.imsize[1],ch))
        self.ch = ch
        
        if ch == 2:
            if loader.color == 'CIE':
                self.startCh = 1
                self.endCh = 3
            elif loader.color == 'YUV':
                self.startCh = 1
                self.endCh = 3
            elif loader.color == 'HSV':
                self.startCh = 0
                self.endCh = 2
        else:
            self.startCh = 0
            self.endCh = 3

    def __str__(self):
        return f"Still Background estimator with {len(self.loader)} images"
        
    def fit(self) -> None:
        all_img = np.empty(
            (self.imsize[0], self.imsize[1], self.ch, len(self.loader)),
            dtype=float
        )
        for ii, (img_id, img) in tqdm(enumerate(self.loader), desc="Fit progress"):
            img = np.array(img)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img = img.astype(float).flatten()
            all_img[:,:,:, ii] = img[:,:,self.startCh:self.endCh]

        # self.mean = all_img.mean(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        # self.variance = all_img.var(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        self.mean = all_img.mean(axis=3)
        self.variance = all_img.var(axis=3)

    def predict(self, img: np.ndarray):
        img = img.astype(float)
        maskCh = np.abs(img - self.mean) > (self.tol * (np.sqrt(self.variance) + 2))
        mask = np.ones((self.imsize[0],self.imsize[1]))
        # mask = np.zeros((self.imsize[0],self.imsize[1]))
        
        for i in range(self.ch):
           mask *= maskCh[:,:,i]
           # mask += maskCh[:,:,i]
            
        # FIXME: maybe not the fastest way
        # calculate new mean and variance for all pixels
        new_mean = self.rho * img + (1 - self.rho) * self.mean
        new_variance = self.rho * (img - self.mean) ** 2 + (1 - self.rho) * self.variance
        
        # replace only if it is background
        self.mean = np.where(np.swapaxes(np.swapaxes(np.tile(mask,(self.ch,1,1)),0,1),1,2) == 0, new_mean, self.mean)
        self.variance = np.where(np.swapaxes(np.swapaxes(np.tile(mask,(self.ch,1,1)),0,1),1,2) == 0, new_variance, self.variance)
        
        return mask

    def save_estimator(self, out_path: Path) -> None:
        np.savez(
            str(out_path),
            mean=self.mean,
            variance=self.variance
        )

    def set_tol(self, tol: float) -> None:
        self.tol = tol

    def set_rho(self, rho: float) -> None:
        self.rho = rho

    def load_estimator(self, in_path: Path) -> None:
        npz_arr = np.load(str(in_path))
        self.mean = npz_arr["mean"]
        self.variance = npz_arr["variance"]

    def viz_estimator(self):
        ax = plt.subplot()
        plt.title("Background Estimator Standard Deviation")
        im = ax.imshow(np.sqrt(self.variance), cmap="hot", interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.figure(dpi=150)

        plt.show()
        plt.close()


class AdaptativeEstimatorColour:
    def __init__(self, loader: FrameLoader, tol: float = 2.5, rho: float = 0.5) -> None:
        self.loader = loader
        self.tol = tol
        self.imsize = self.loader.probe()

        self.mean = np.zeros(self.imsize[:2])
        self.variance = np.zeros(self.imsize[:2])
        self.rho = rho

    def __str__(self):
        return f"Still Background estimator with {len(self.loader)} images"

    def fit(self) -> None:
        all_img = np.empty(
            (self.imsize[0] * self.imsize[1], len(self.loader)),
            dtype=float
        )
        for ii, img in tqdm(enumerate(self.loader), desc="Fit progress"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(float).flatten()
            all_img[:, ii] = img

        self.mean = all_img.mean(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        self.variance = all_img.var(axis=-1).reshape((self.imsize[0], self.imsize[1]))

    def predict(self, img: np.ndarray):
        img = img.astype(float)
        mask = np.abs(img - self.mean) > (self.tol * (np.sqrt(self.variance) + 2))

        # FIXME: maybe not the fastest way
        # calculate new mean and variance for all pixels
        new_mean = self.rho * img + (1 - self.rho) * self.mean
        new_variance = np.sqrt(self.rho * (img - self.mean) ** 2 + (1 - self.rho) * self.variance ** 2)

        # replace only if it is background
        self.mean = np.where(mask is False, new_mean, self.mean)
        self.variance = np.where(mask is False, new_variance, self.variance)

        return mask

    def save_estimator(self, out_path: Path) -> None:
        np.savez(
            str(out_path),
            mean=self.mean,
            variance=self.variance
        )

    def set_tol(self, tol: float) -> None:
        self.tol = tol

    def set_rho(self, rho: float) -> None:
        self.rho = rho

    def load_estimator(self, in_path: Path) -> None:
        npz_arr = np.load(str(in_path))
        self.mean = npz_arr["mean"]
        self.variance = npz_arr["variance"]

    def viz_estimator(self):
        ax = plt.subplot()
        plt.title("Background Estimator Standard Deviation")
        im = ax.imshow(np.sqrt(self.variance), cmap="hot", interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.figure(dpi=150)

        plt.show()
        plt.close()

def cleanup_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    kern = np.ones((ksize, ksize))
    mask = mask.astype(np.uint8)
    mask = cv2.medianBlur(mask, ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    return mask


def get_bboxes(mask: np.ndarray, lowbound: int) -> List[Bbox]:
    num, cc = cv2.connectedComponents(mask)
    bboxes = []

    for ii in range(1, num):
        indices = np.where(cc == ii)

        minx = np.min(indices[1])
        maxx = np.max(indices[1])
        miny = np.min(indices[0])
        maxy = np.max(indices[0])

        width = maxx - minx
        height = maxy - miny

        if width * height >= lowbound:
            bboxes.append((
                int(minx), int(miny), int(width), int(height)
            ))

    return bboxes

import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm.auto import tqdm

from data import FrameLoader


class StillBackgroundEstimatorGrayscale:
    def __init__(self, loader: FrameLoader) -> None:
        self.loader = loader
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
        for ii, img in tqdm(enumerate(self.loader), desc="Fit progress"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(float).flatten()
            all_img[:, ii] = img

        self.mean = all_img.mean(axis=-1).reshape((self.imsize[0], self.imsize[1]))
        self.variance = all_img.var(axis=-1).reshape((self.imsize[0], self.imsize[1]))

    def predict(self, img: np.ndarray):
        img = img.astype(float)
        mask = np.abs(img - self.mean) > self.variance

        return mask

    def save_estimator(self, out_path: Path) -> None:
        np.savez(
            str(out_path),
            mean=self.mean,
            variance=self.variance
        )

    def load_estimator(self, in_path: Path) -> None:
        npz_arr = np.load(str(in_path))
        self.mean = npz_arr["mean"]
        self.variance = npz_arr["variance"]

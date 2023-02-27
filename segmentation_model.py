import cv2
import numpy as np

import utils


class SegmentationModel(object):
    def __init__(self, name: str, img_ds: int = 1):
        self.name = name
        self.img_ds = img_ds

    def predict(self, image: np.array) -> np.array:
        return np.zeros_like(image)

    def prepare_output(self, out_labels: np.array, imh: int, imw: int):
        if self.img_ds == 1:
            out_labels = np.reshape(out_labels, newshape=(imh, imw))
        else:
            out_labels = np.reshape(out_labels, newshape=(imh // self.img_ds, imw // self.img_ds))
            out_labels = cv2.resize(out_labels, (imw, imh), interpolation=cv2.INTER_AREA)
        return utils.lbl2oh_mult(out_labels, N_CLASS=3)

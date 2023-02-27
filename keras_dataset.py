import math

import os
import cv2
import numpy as np
from keras.utils.data_utils import Sequence
from keras_utils import lbl2oh_mult, lbl2oh_mult_with_bg, lbl2oh_mult2bin_with_bg

from albumentations import (
    Blur, GaussNoise, MotionBlur, MedianBlur,
    RandomBrightnessContrast, OneOf, Compose, Sharpen, Emboss
)


def _augm(p=.5):
    return Compose([
        OneOf([
            GaussNoise(),
        ], p=0.33),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.33),
        OneOf([
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.33),
    ], p=p)


augm_train = _augm(1)
augm_val = _augm(1)


class Dataset(Sequence):
    def __init__(self,
                 database_path,
                 images_num_on_axes,
                 window_size,
                 batch_size,
                 is_binary=False,
                 with_background=False,
                 transforms=None,
                 shuffle=True):
        super(Dataset, self).__init__()
        self.is_binary = is_binary
        self.with_background = with_background
        self.transforms = transforms
        self.shuffle = shuffle
        self.indexes, self.data_map, self.label_map = self.get_images_and_labels(database_path, window_size, images_num_on_axes)
        print(f'dataset_len = {len(self.indexes)}, batch_size = {batch_size}, dataset_len mod batch_size = {len(self.indexes) % batch_size}')
        if batch_size is None:
            self.batch_size = len(self.indexes)
        else:
            self.batch_size = batch_size

    @staticmethod
    def hash_key(video_num, sample_num, image_num):
        return video_num * 100_00 + sample_num * 100 + image_num

    def get_images_and_labels(self, database_path, window_size, images_num_on_axes):
        indexes = []
        data_map = {}
        label_map = {}
        images_dir = os.path.join(database_path, 'images')
        labels_dir = os.path.join(database_path, 'labels')
        if type(window_size) is tuple:
            wy = window_size[0]
            wx = window_size[1]
        else:
            wy = wx = window_size
        for video_num in range(0, len(os.listdir(images_dir))):
            img_video_path = os.path.join(images_dir, str(video_num))
            lbl_video_path = os.path.join(labels_dir, str(video_num))
            for sample_num in range(0, len(os.listdir(img_video_path))):
                image = cv2.imread(os.path.join(img_video_path, str(sample_num) + '.jpg'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = np.load(os.path.join(lbl_video_path, str(sample_num) + '.npy'))
                if self.with_background:
                    one_hot_label = lbl2oh_mult_with_bg(label, 3)
                else:
                    one_hot_label = lbl2oh_mult(label, 3)
                if self.is_binary:
                    one_hot_label = lbl2oh_mult2bin_with_bg(label)
                image_num = 0
                dy = (image.shape[0] - wy) // max((images_num_on_axes[0] - 1), 1)
                dx = (image.shape[1] - wx) // max((images_num_on_axes[1] - 1), 1)
                for row_num in range(0, images_num_on_axes[0]):
                    for column_num in range(0, images_num_on_axes[1]):
                        indexes.append(Dataset.hash_key(video_num, sample_num, image_num))
                        y0, y1 = row_num * dy, row_num * dy + wy
                        x0, x1 = column_num * dx, column_num * dx + wx
                        data_map[Dataset.hash_key(video_num, sample_num, image_num)] = image[y0:y1, x0:x1]
                        label_map[Dataset.hash_key(video_num, sample_num, image_num)] = one_hot_label[y0:y1, x0:x1]
                        image_num += 1
        return np.array(indexes), data_map, label_map

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.indexes[idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.indexes))]
        images, labels = [], []
        for key in batch:
            image = self.data_map[key].copy()
            label = self.label_map[key]
            if self.transforms is not None:
                image = self.transforms(image=image)['image']
            images.append((np.asarray(image, np.float32) / 255.0))
            labels.append(label)
        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class ValidationDataset(Sequence):
    def __init__(self, dataset_path, with_background=False, is_binary=False):
        super(ValidationDataset, self).__init__()
        self.with_background = with_background
        self.is_binary = is_binary
        self.indexes, self.data_map, self.label_map = self.get_images_and_labels(dataset_path)

    @staticmethod
    def hash_key(video_num, sample_num):
        return video_num * 1000 + sample_num

    @staticmethod
    def unhash(hash_key):
        return hash_key // 1000, hash_key % 1000

    def get_images_and_labels(self, database_path):
        indexes = []
        data_map = {}
        label_map = {}
        images_dir = os.path.join(database_path, 'images')
        labels_dir = os.path.join(database_path, 'labels')
        for video_num in range(0, len(os.listdir(images_dir))):
            img_video_path = os.path.join(images_dir, str(video_num))
            lbl_video_path = os.path.join(labels_dir, str(video_num))
            for sample_num in range(0, len(os.listdir(img_video_path))):
                image = cv2.imread(os.path.join(img_video_path, str(sample_num) + '.jpg'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = np.load(os.path.join(lbl_video_path, str(sample_num) + '.npy'))
                if self.with_background:
                    one_hot_label = lbl2oh_mult_with_bg(label, 3)
                else:
                    one_hot_label = lbl2oh_mult(label, 3)
                if self.is_binary:
                    one_hot_label = lbl2oh_mult2bin_with_bg(label)
                data_map[ValidationDataset.hash_key(video_num, sample_num)] = image
                label_map[ValidationDataset.hash_key(video_num, sample_num)] = one_hot_label
                indexes.append(ValidationDataset.hash_key(video_num, sample_num))
        return np.array(indexes), data_map, label_map

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        key = self.indexes[idx]
        video_num, sample_num = self.unhash(key)
        image = np.asarray(self.data_map[key].copy(), dtype=np.float32) / 255.0
        label = self.label_map[key]
        return np.array(image), np.array(label), video_num, sample_num

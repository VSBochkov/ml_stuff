import os.path
from os.path import join

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

import utils
from ml_models.prepare_input_features import calculate_input_features
from segmentation_model import SegmentationModel
from snapshots import ML_SNAPSHOTS_DIR


class OSDecisionTree(SegmentationModel):
    def __init__(self, name: str,
                 train_x: pd.DataFrame,
                 train_y: pd.DataFrame,
                 img_ds: int,
                 **dtree_args):
        super().__init__(name, img_ds)
        self.num_classes = train_y.max()
        if os.path.exists(join(ML_SNAPSHOTS_DIR, f'{name}.dtree')):
            self.nnc = joblib.load(join(ML_SNAPSHOTS_DIR, f'{name}.joblib'))
        else:
            self.nnc = DecisionTreeClassifier(**dtree_args).fit(train_x, train_y)
            joblib.dump(self.nnc, join(ML_SNAPSHOTS_DIR, f'{name}.joblib'))

    def predict(self, anim: np.array) -> np.array:
        images_num, imh, imw = anim.shape[0], anim.shape[1], anim.shape[2]
        features = calculate_input_features(anim, ds=self.img_ds)
        output = self.nnc.predict(features)
        return self.prepare_output(output, imh, imw)

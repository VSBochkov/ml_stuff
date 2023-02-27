import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import utils
from ml_models.prepare_input_features import calculate_input_features
from segmentation_model import SegmentationModel


class OneShotKNN(SegmentationModel):
    def __init__(self, name: str,
                 num_neighb: int,
                 weights_type: str,
                 train_x: pd.DataFrame,
                 train_y: pd.DataFrame,
                 img_ds: int = 1):
        super().__init__(name, img_ds)
        self.num_classes = train_y.max()
        self.nnc = KNeighborsClassifier(n_neighbors=num_neighb, weights=weights_type)
        self.nnc.fit(train_x, train_y)

    def predict(self, anim: np.array) -> np.array:
        images_num, imh, imw = anim.shape[0], anim.shape[1], anim.shape[2]
        features = calculate_input_features(anim, ds=self.img_ds)
        output = self.nnc.predict(features)
        return self.prepare_output(output, imh, imw)

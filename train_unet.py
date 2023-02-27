import os.path

import keras.optimizers

from keras_unet_models import unet
from keras_utils import jaccard_acc, bce_jaccard_loss
from keras_dataset import Dataset, augm_train, augm_val

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

WITH_BG = False
IS_LIGHT = False
dilation_rate = (1, 1)
AUGMENTATION_ENABLED = True
# COMB_TYPE = 'fs'
# IMAGES_AXES = (1, 3)
INPUT_SIZE = 224
BS_TRAIN = 120 if IS_LIGHT else 64
COMB_TYPE = 'ow'
IMAGES_AXES = (1, 1)
#INPUT_SIZE = 304
#BS_TRAIN = 30 if IS_LIGHT else 15
# INPUT_SIZE = 336
# BS_TRAIN = 24 if IS_LIGHT else 12
BS_VAL = 64
LEARN_RATES = {
    0:  1e-3,
    50: 1e-4,
    100: 1e-5,
    150: 1e-6,
    200: 1e-7
}
#LEARN_RATES = {0: 1e-7}

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4
EPOCH_N = 250
TOTAL_EPOCHS = 2000

DATASET_PATH = '../dataset/{}_{}'.format(COMB_TYPE, INPUT_SIZE)
MODEL_NAME = 'unet_bn'
if WITH_BG:
    MODEL_NAME += '_bg'
MODEL_NAME += f'_{INPUT_SIZE}_{COMB_TYPE}'
OUTPUT_PATH = '../output/keras/{}'.format(MODEL_NAME)


if __name__ == '__main__':
    trainDataset = Dataset(
        os.path.join(DATASET_PATH, "train"),
        images_num_on_axes=IMAGES_AXES,
        window_size=INPUT_SIZE,
        batch_size=BS_TRAIN,
        transforms=augm_train,
        with_background=WITH_BG
    )

    bounds = []
    rates = []

    for iteration in range(int(TOTAL_EPOCHS / EPOCH_N)):
        for epoch_num in list(LEARN_RATES.keys()):
            bounds.append((iteration * EPOCH_N + epoch_num) * len(trainDataset))
        for learn_rate in LEARN_RATES.values():
            rates.append(learn_rate)
    bounds = bounds[1:]

    lr_schedule = keras.optimizer_v2.learning_rate_schedule.PiecewiseConstantDecay(
        boundaries=bounds,
        values=rates
    )

    optimizer = keras.optimizers.adam_v2.Adam(
        learning_rate=lr_schedule,
        beta_1=BETAS[0],
        beta_2=BETAS[1],
        decay=WEIGHT_DECAY
    )

    n_class = 4 if WITH_BG else 3
    model = unet(n_class=n_class, input_size=INPUT_SIZE, batch_size=None, width_div=2 if IS_LIGHT else 1, use_batch_norm=True, dilation_rate=dilation_rate)
    model.compile(
        optimizer=optimizer,
        loss=bce_jaccard_loss,
        metrics=jaccard_acc
    )

    training_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_PATH + '/snapshots/best_val_loss_{epoch}.hdf5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_PATH + '/snapshots/best_val_acc_{epoch}.hdf5',
            monitor='val_jaccard_acc',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=OUTPUT_PATH + '/logdir')
    ]

    history = model.fit(
        trainDataset,
        validation_data=Dataset(
            os.path.join(DATASET_PATH, "test"),
            images_num_on_axes=IMAGES_AXES,
            window_size=INPUT_SIZE,
            batch_size=BS_VAL,
            transforms=augm_val,
            with_background=WITH_BG
        ),
        epochs=TOTAL_EPOCHS,
        callbacks=training_callbacks
    )

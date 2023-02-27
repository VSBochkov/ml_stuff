import sys

import numpy as np

from keras_unet_models import unet, wuunet, bmunet
from keras_deeplab_models import deeplab, bmdeeplab
from keras_infered_models import infered_model, infered_fs_model

import cv2
import time
import os

in_video_dir = '/home/vbochkov/workspace/development/fire_detection/fire_videos/'
out_video_dir = '/home/vbochkov/Desktop/'
video = sys.argv[1]
BATCH_SIZE=1

MODEL_NAME = sys.argv[2]
WIN_SIZE = int(sys.argv[3])
IMAGE_WIDTH = WIN_SIZE
IMAGE_HEIGHT = WIN_SIZE
if MODEL_NAME == 'wuunet':
    CHECKPOINT = 'output/keras/wuunet_224_ow/snapshots/best_val_bin_acc_600.hdf5'
    wuunet = wuunet(batch_size=BATCH_SIZE, train=False)
    wuunet.load_weights(CHECKPOINT)
    THRESHOLD = 0.61958
    model = infered_model(model=wuunet, threshold=THRESHOLD)
elif MODEL_NAME == 'wuunet_light':
    CHECKPOINT = 'output/keras/wuunet_light_224_ow/snapshots/best_val_mult_acc_792.hdf5'
    wuunet_light = wuunet(batch_size=BATCH_SIZE, light=True, train=False)
    wuunet_light.load_weights(CHECKPOINT)
    THRESHOLD = 0.25
    model = infered_model(model=wuunet_light, threshold=THRESHOLD)
elif MODEL_NAME == 'unet':
    CHECKPOINT = 'output/keras/unet_224_ow/snapshots/best_val_acc_789.hdf5'
    unet = unet(batch_size=BATCH_SIZE)
    unet.load_weights(CHECKPOINT)
    THRESHOLD = 0.5594500000000001
    model = infered_model(model=unet, threshold=THRESHOLD)
elif MODEL_NAME == 'bmunet':
    CHECKPOINT = 'output/keras/bmunet_224_ow/snapshots/best_val_bin_acc_545.hdf5'
    bmunet = bmunet(batch_size=BATCH_SIZE, train=False)
    bmunet.load_weights(CHECKPOINT)
    THRESHOLD = 0.5073799999999999
    model = infered_model(model=bmunet, threshold=THRESHOLD)
elif MODEL_NAME == 'bmunet_light':
    CHECKPOINT = 'output/keras/bmunet_light_304_ow/snapshots/best_val_bin_acc_1316.hdf5'
    bmunet = bmunet(batch_size=BATCH_SIZE, input_size=WIN_SIZE, light=True, train=False)
    bmunet.load_weights(CHECKPOINT)
    THRESHOLD = 0.43923
    model = infered_model(model=bmunet, threshold=THRESHOLD)
elif MODEL_NAME == 'deeplab':
    CHECKPOINT = 'output/keras/deeplab_224_ow/snapshots/best_val_loss_529.hdf5'
    deeplab = deeplab(batch_size=BATCH_SIZE)
    deeplab.load_weights(CHECKPOINT)
    THRESHOLD = 0.8894299999999999
    model = infered_model(model=deeplab, threshold=THRESHOLD)
elif MODEL_NAME == 'bmdeeplab':
    CHECKPOINT = 'output/keras/bmdeeplab_224_ow/snapshots/best_val_mult_acc_347.hdf5'
    bmdeeplab = bmdeeplab(batch_size=BATCH_SIZE, train=False)
    bmdeeplab.load_weights(CHECKPOINT)
    THRESHOLD = 0.6818200000000001
    model = infered_model(model=bmdeeplab, threshold=THRESHOLD)

# Full-size 224
# bmunet
# IMAGE_WIDTH = 428
# IMAGE_HEIGHT = 224
# MODEL_NAME = 'bmunet'
# CHECKPOINT = 'output/keras/bmunet_light_224_fs/snapshots/best_val_mult_acc_91.hdf5'
# bmunet = bmunet(batch_size=BATCH_SIZE * 2, light=True, train=False)
# bmunet.load_weights(CHECKPOINT)
# THRESHOLD = 0.52006
# model = infered_fs_model(model=bmunet, threshold=THRESHOLD, image_width=IMAGE_WIDTH, win_size=IMAGE_HEIGHT)


def get_overlay(image, signal):
    imh, imw = IMAGE_HEIGHT, IMAGE_WIDTH
    overlay = np.ones((imh * 2, imw, 3), dtype=np.uint8) * 0xff
    red = np.array([0x0, 0x00, 0xff])
    orange = np.array([0x00, 0xc8, 0xff])
    yellow = np.array([0x00, 0xff, 0xff])
    for i in range(0, imh):
        for j in range(0, imw):
            if signal[i, j] == 0:
                overlay[i, j] = image[i, j]
            elif signal[i, j] == 1:
                overlay[i, j] = red
                overlay[i + imh, j] = red
            elif signal[i, j] == 2:
                overlay[i, j] = orange
                overlay[i + imh, j] = orange
            elif signal[i, j] == 3:
                overlay[i, j] = yellow
                overlay[i + imh, j] = yellow
    cv2.putText(overlay, MODEL_NAME, (10, imh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return overlay


def inference():
    cap = cv2.VideoCapture(os.path.join(in_video_dir, video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(video)
    it = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(out_video_dir, MODEL_NAME + '_' + video), fourcc, fps, (IMAGE_WIDTH, IMAGE_HEIGHT * 2))
    while True:
        t1 = time.process_time()
        ok, orig_img = cap.read()
        if not ok:
            break
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        t2 = time.process_time()
        img = cv2.resize(img, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT))
        out = model.predict(np.expand_dims(img, axis=0))
        out = np.squeeze(out)
        t3 = time.process_time()
        print('----------------------------')
        print('{}, {}, {}, FPT = {}'.format(video, MODEL_NAME, it, t3 - t1))
        print('frame grabbing time = {}'.format(t2 - t1))
        print('model inference time = {}'.format(t3 - t2))
        out_video.write(get_overlay(img, out))
        it += 1
    cap.release()
    out_video.release()


if __name__ == '__main__':
    inference()

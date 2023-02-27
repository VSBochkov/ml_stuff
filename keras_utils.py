import sys
from os.path import join

import numpy as np
from focal_loss import binary_focal_loss

from tensorflow.keras import backend as K


def lbl2oh_mult(label_map, N_CLASS):
    mc_label = np.zeros((label_map.shape[0], label_map.shape[1], N_CLASS), dtype=np.bool)
    for i in range(1, N_CLASS + 1):
        mc_label[:, :, i - 1] = label_map == i
    return np.asarray(mc_label, dtype=np.float32)


def lbl2oh_mult_with_bg(label_map, N_CLASS):
    mc_label = np.zeros((label_map.shape[0], label_map.shape[1], N_CLASS + 1), dtype=np.bool)
    for i in range(0, N_CLASS + 1):
        mc_label[:, :, i] = label_map == i
    return np.asarray(mc_label, dtype=np.float32)


def lbl2oh_mult2bin(label_map):
    bin_label = np.zeros((label_map.shape[0], label_map.shape[1], 1), dtype=np.bool)
    max_val = min(np.max(label_map), 3)
    for i in range(1, max_val + 1):
        bin_label[:, :, 0] = np.logical_or(bin_label, label_map == i)
    return np.asarray(bin_label, dtype=np.float32)


def lbl2oh_mult2bin_with_bg(label_map):
    bin_label = np.zeros((label_map.shape[0], label_map.shape[1], 2), dtype=np.bool)
    bin_label[:, :, 0] = np.asarray(label_map == 0)
    bin_label[:, :, 1] = np.asarray(label_map > 0)
    return np.asarray(bin_label, dtype=np.float32)


def oh_mult2bin(mult_label_map):
    bin_label = np.zeros((mult_label_map.shape[0], mult_label_map.shape[1], 1), dtype=np.float32)
    bin_label[:, :, 0] = np.max(mult_label_map, axis=2)
    return bin_label


def oh2lbl_mult(mult_one_hot):
    max = np.amax(mult_one_hot, axis=2, keepdims=False)
    argmax = np.argmax(mult_one_hot, axis=2)
    return argmax + np.asarray(max > 0, dtype=np.int64)


def oh_with_bg2lbl_mult(mult_one_hot):
    return np.argmax(mult_one_hot, axis=2)


def oh2oh_with_bg_mult(mult_one_hot):
    return lbl2oh_mult_with_bg(oh2lbl_mult(mult_one_hot), mult_one_hot.shape[2])


def oh_mult2bin_with_bg(mult_label_map):
    bin_label = np.zeros((mult_label_map.shape[0], mult_label_map.shape[1], 2), dtype=np.float32)
    bin_label[:, :, 0] = mult_label_map[:, :, 0]
    bin_label[:, :, 1] = np.max(mult_label_map[:, :, 1:], axis=2)
    return bin_label


def np_dice_acc(pred, target, smooth=1.):
    pred = np.ascontiguousarray(pred)
    target = np.ascontiguousarray(target)
    intersection = (pred * target).sum(axis=(1, 2))
    return (2. * intersection + smooth) / (pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2)) + smooth)


def np_jaccard_acc(pred, target, smooth=1.):
    axes = (0, 1)
    pred = np.ascontiguousarray(pred)
    target = np.ascontiguousarray(target)
    intersection = (pred * target).sum(axis=axes)
    return (intersection + smooth) / (pred.sum(axis=axes) + target.sum(axis=axes) - intersection + smooth)


def jaccard_acc(y_true, y_pred, smooth=1.):
    axes = [1, 2]
    intersection = K.sum(y_true * y_pred, axis=axes)
    sum_ = K.sum(y_true + y_pred, axis=axes)
    return (intersection + smooth) / (sum_ - intersection + smooth)


def bin_jaccard_acc(y_true, y_pred, smooth=1.):
    return jaccard_acc(K.max(y_true, axis=3, keepdims=True), y_pred, smooth)


def bin_jaccard_acc_with_bg(y_true, y_pred, smooth=1.):
    if y_true.shape[1] is None:
        bin_y_true = K.zeros_like(y_pred)
    else:
        bin_y_true = oh_mult2bin_with_bg(y_true)
    return jaccard_acc(bin_y_true, y_pred, smooth)


def soft_jaccard_loss(y_true, y_pred):
    return -K.log(jaccard_acc(y_true, y_pred, 1.0))


def jaccard_loss(y_true, y_pred):
    return K.mean(soft_jaccard_loss(y_true, y_pred))


def bce_jaccard_loss(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.mean(bce) + K.mean(soft_jaccard_loss(y_true, y_pred))


def bce_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def focal_loss(y_true, y_pred):
    return K.mean(binary_focal_loss(y_true, y_pred, gamma=2.0))


def focal_jaccard_loss(y_true, y_pred):
    focal = binary_focal_loss(y_true, y_pred, gamma=2.0)
    return K.mean(focal) + K.mean(soft_jaccard_loss(y_true, y_pred))


def bce_bin_jaccard_loss(y_true, y_pred):
    bin_y_true = K.max(y_true, axis=3, keepdims=True)
    bce = K.binary_crossentropy(bin_y_true, y_pred)
    return K.mean(bce) + K.mean(soft_jaccard_loss(bin_y_true, y_pred))


def bce_bin_jaccard_loss_with_bg(y_true, y_pred):
    if y_true.shape[1] is None:
        bin_y_true = K.zeros_like(y_pred)
    else:
        bin_y_true = oh_mult2bin_with_bg(y_true)
    bce = K.binary_crossentropy(bin_y_true, y_pred)
    return K.mean(bce) + K.mean(soft_jaccard_loss(bin_y_true, y_pred))


def bin_focal_jaccard_loss_with_bg(y_true, y_pred):
    if y_true.shape[1] is None:
        bin_y_true = K.zeros_like(y_pred)
    else:
        bin_y_true = oh_mult2bin_with_bg(y_true)
    focal = binary_focal_loss(bin_y_true, y_pred, gamma=2.0)
    return K.mean(focal) + K.mean(soft_jaccard_loss(bin_y_true, y_pred))


def bin_mul_bce_jaccard_loss_with_bg(mul_y_true, mul_y_pred):
    if mul_y_true.shape[1] is None:
        return bce_jaccard_loss(mul_y_true, mul_y_pred)
    else:
        bin_y_true = oh_mult2bin_with_bg(mul_y_true)
        bin_y_pred = oh_mult2bin_with_bg(mul_y_pred)
        return bce_jaccard_loss(mul_y_true, mul_y_pred) + bce_jaccard_loss(bin_y_true, bin_y_pred)


def bin_mul_jaccard_loss_with_bg(mul_y_true, mul_y_pred):
    if mul_y_true.shape[1] is None:
        return jaccard_loss(mul_y_true, mul_y_pred)
    else:
        bin_y_true = oh_mult2bin_with_bg(mul_y_true)
        bin_y_pred = oh_mult2bin_with_bg(mul_y_pred)
        return jaccard_loss(mul_y_true, mul_y_pred) + jaccard_loss(bin_y_true, bin_y_pred)


def bin_mul_focal_jaccard_loss_with_bg(mul_y_true, mul_y_pred):
    if mul_y_true.shape[1] is None:
        return focal_jaccard_loss(mul_y_true, mul_y_pred)
    else:
        bin_y_true = oh_mult2bin_with_bg(mul_y_true)
        bin_y_pred = oh_mult2bin_with_bg(mul_y_pred)
        return focal_jaccard_loss(mul_y_true, mul_y_pred) + focal_jaccard_loss(bin_y_true, bin_y_pred)


if __name__ == '__main__':
    mult_labels = np.load(join('..', 'dataset', 'ow_304', 'test', 'labels', '0', '0.npy'))
    _max, _min = mult_labels.max(), mult_labels.min()
    mult_oh = lbl2oh_mult(mult_labels, 3)
    mult_oh_with_bg = oh2oh_with_bg_mult(mult_oh)
    result_mult_labels = oh_with_bg2lbl_mult(mult_oh_with_bg)
    result_max, result_min = result_mult_labels.max(), result_mult_labels.min()
    are_eq = np.equal(mult_labels, result_mult_labels)
    print(f'are_eq = {are_eq}')

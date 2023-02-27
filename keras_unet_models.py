from keras.models import *
from keras.layers import *


def double_conv(inputs, out_channels, use_batch_norm=False, dilation_rate=(1, 1)):
    kernel_size = 3
    c1 = Conv2D(out_channels, kernel_size, padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(inputs)
    if use_batch_norm:
        c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    c2 = Conv2D(out_channels, kernel_size, padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(c1)
    if use_batch_norm:
        c2 = BatchNormalization()(c2)
    return ReLU()(c2)


def unet(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)

    return Model(inputs=inputs, outputs=conv_last)


def unet3(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    bottleneck = double_conv(pool2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)
    return Model(inputs=inputs, outputs=conv_last)


def unet2(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    bottleneck = double_conv(pool1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)
    return Model(inputs=inputs, outputs=conv_last)


def unet5(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    dconv_down4 = double_conv(pool3, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    pool4 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down4)
    bottleneck = double_conv(pool4, width * 16, use_batch_norm, dilation_rate=dilation_rate)
    upsampled4 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge4 = concatenate([upsampled4, dconv_down4], axis=3)
    dconv_up4 = double_conv(merge4, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up4)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)

    return Model(inputs=inputs, outputs=conv_last)


def bmunet(n_class, batch_size, input_size, width_div: int, with_bg=False, use_batch_norm=False, dilation_rate=(1, 1), train=True):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, use_batch_norm, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    out_bin = Conv2D(2 if with_bg else 1, kernel_size=1, activation='sigmoid', padding='same',
                     dilation_rate=dilation_rate, name='out_bin')(dconv_up1)
    merge_img_bin_conv = concatenate([out_bin, dconv_up1], axis=3)
    conv_up1_1 = double_conv(merge_img_bin_conv, width // 2, use_batch_norm, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, name='out_mult')(conv_up1_1)

    if train:
        return Model(
            inputs=inputs,
            outputs={
                'out_bin': out_bin,
                'out_mult': out_mult
            }
        )
    else:
        return Model(inputs=inputs, outputs=out_mult)


def old_bmunet(n_class=3, batch_size=None, input_size=224, light=False, dilation_rate=(1, 1), train=True):
    width = 32 if light else 64
    updown_sampling_size = (2, 2)
    input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width * 4, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width * 8, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width * 4, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, dilation_rate=dilation_rate)
    out_bin = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, name='out_bin')(dconv_up1)

    merge_img_bin_conv = concatenate([dconv_down1, out_bin, dconv_up1], axis=3)
    conv_up1_1 = double_conv(merge_img_bin_conv, width, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, name='out_mult')(conv_up1_1)

    if train:
        return Model(
            inputs=inputs,
            outputs={
                'out_bin': out_bin,
                'out_mult': out_mult
            }
        )
    else:
        return Model(inputs=inputs, outputs=out_mult)


def uunet(n_class, batch_size, input_size, width_div: int, with_bg=False, use_batch_norm=True, dilation_rate=(1, 1), train=True):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'

    inputs = Input(input_shape, batch_size=batch_size)

    dconv_down1_1 = double_conv(inputs, width, use_batch_norm, dilation_rate=dilation_rate)
    pool1_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1_1)
    dconv_down2_1 = double_conv(pool1_1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2_1)
    dconv_down3_1 = double_conv(pool2_1, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3_1)
    bottleneck_1 = double_conv(pool3_1, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck_1)
    merge3_1 = concatenate([upsampled3_1, dconv_down3_1], axis=3)
    dconv_up3_1 = double_conv(merge3_1, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3_1)
    merge2_1 = concatenate([upsampled2_1, dconv_down2_1], axis=3)
    dconv_up2_1 = double_conv(merge2_1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2_1)
    merge1_1 = concatenate([upsampled1_1, dconv_down1_1], axis=3)
    dconv_up1_1 = double_conv(merge1_1, width, use_batch_norm, dilation_rate=dilation_rate)
    out_bin = Conv2D(2 if with_bg else 1, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', name='out_bin')(dconv_up1_1)

    inputs_bin = concatenate([inputs, out_bin], axis=3)

    dconv_down1_2 = double_conv(inputs_bin, width, dilation_rate=dilation_rate)
    pool1_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1_2)
    merge_enc1_2 = concatenate([pool1_2, dconv_up2_1], axis=3)
    dconv_down2_2 = double_conv(merge_enc1_2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2_2)
    merge_enc2_2 = concatenate([pool2_2, dconv_up3_1], axis=3)
    dconv_down3_2 = double_conv(merge_enc2_2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3_2)
    merge_enc3_2 = concatenate([pool3_2, bottleneck_1], axis=3)
    bottleneck_2 = double_conv(merge_enc3_2, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck_2)
    merge_dec3_2 = concatenate([upsampled3_2, dconv_down3_2])
    dconv_up3_2 = double_conv(merge_dec3_2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3_2)
    merge_dec2_2 = concatenate([upsampled2_2, dconv_down2_2])
    dconv_up2_2 = double_conv(merge_dec2_2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2_2)
    merge_dec1_2 = concatenate([upsampled1_2, dconv_down1_2], axis=3)
    dconv_up1_2 = double_conv(merge_dec1_2, width, use_batch_norm, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', name='out_mult')(dconv_up1_2)

    if train:
        return Model(
            inputs=inputs,
            outputs={
                'out_bin': out_bin,
                'out_mult': out_mult
            }
        )
    else:
        return Model(inputs=inputs, outputs=out_mult)


def wuunet(n_class, batch_size, input_size, width_div: int, with_bg=False, use_batch_norm=False, dilation_rate=(1, 1), train=True):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'

    inputs = Input(input_shape, batch_size=batch_size)

    dconv_down1_1 = double_conv(inputs, width, use_batch_norm, dilation_rate=dilation_rate)
    pool1_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1_1)
    dconv_down2_1 = double_conv(pool1_1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2_1)
    dconv_down3_1 = double_conv(pool2_1, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3_1)
    bottleneck_1 = double_conv(pool3_1, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck_1)
    merge3_1 = concatenate([upsampled3_1, dconv_down3_1], axis=3)
    dconv_up3_1 = double_conv(merge3_1, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3_1)
    merge2_1 = concatenate([upsampled2_1, dconv_down2_1], axis=3)
    dconv_up2_1 = double_conv(merge2_1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2_1)
    merge1_1 = concatenate([upsampled1_1, dconv_down1_1], axis=3)
    dconv_up1_1 = double_conv(merge1_1, width, use_batch_norm, dilation_rate=dilation_rate)
    out_bin = Conv2D(2 if with_bg else 1, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', name='out_bin')(dconv_up1_1)

    inputs_bin = concatenate([inputs, out_bin], axis=3)

    dconv_down1_2 = double_conv(inputs_bin, width, use_batch_norm, dilation_rate=dilation_rate)
    pool1_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1_2)
    merge_enc1_2 = concatenate([pool1_2, dconv_down2_1, dconv_up2_1], axis=3)
    dconv_down2_2 = double_conv(merge_enc1_2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2_2)
    merge_enc2_2 = concatenate([pool2_2, dconv_down3_1, dconv_up3_1], axis=3)
    dconv_down3_2 = double_conv(merge_enc2_2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3_2)
    merge_enc3_2 = concatenate([pool3_2, bottleneck_1], axis=3)
    bottleneck_2 = double_conv(merge_enc3_2, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck_2)
    merge_dec3_2 = concatenate([upsampled3_2, dconv_down3_1, dconv_up3_1, dconv_down3_2])
    dconv_up3_2 = double_conv(merge_dec3_2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3_2)
    merge_dec2_2 = concatenate([upsampled2_2, dconv_down2_1, dconv_up2_1, dconv_down2_2])
    dconv_up2_2 = double_conv(merge_dec2_2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2_2)
    merge_dec1_2 = concatenate([upsampled1_2, dconv_down1_1, inputs_bin], axis=3)
    dconv_up1_2 = double_conv(merge_dec1_2, width, use_batch_norm, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', name='out_mult')(dconv_up1_2)

    if train:
        return Model(
            inputs=inputs,
            outputs={
                'out_bin': out_bin,
                'out_mult': out_mult
            }
        )
    else:
        return Model(inputs=inputs, outputs=out_mult)


def unetpp(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1), out_aver_num=3):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, use_batch_norm, dilation_rate=dilation_rate)

    upsampled01 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_down2)
    merge01 = concatenate([dconv_down1, upsampled01], axis=3)
    dconv01 = double_conv(merge01, width, dilation_rate=dilation_rate)
    out01 = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv01)

    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width * 4, use_batch_norm, dilation_rate=dilation_rate)

    upsampled11 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_down3)
    merge11 = concatenate([dconv_down2, upsampled11], axis=3)
    dconv11 = double_conv(merge11, width * 2, dilation_rate=dilation_rate)
    upsampled02 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv11)
    merge02 = concatenate([dconv_down1, dconv01, upsampled02], axis=3)
    dconv02 = double_conv(merge02, width, dilation_rate=dilation_rate)
    out02 = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv02)

    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width * 8, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2, dconv11], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1, dconv01, dconv02], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)
    if out_aver_num == 1:
        return Model(inputs=inputs, outputs=conv_last)
    elif out_aver_num == 2:
        return Model(inputs=inputs, outputs=(conv_last + out02) / 2)
    elif out_aver_num == 3:
        return Model(inputs=inputs, outputs=(conv_last + out02 + out01) / 3)


def unetfpn(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    updown_sampling_size42 = (8, 8)
    updown_sampling_size32 = (4, 4)
    updown_sampling_size22 = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width * 4, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width * 8, use_batch_norm, dilation_rate=dilation_rate)

    bottleneck_2 = double_conv(bottleneck, width / 2, use_batch_norm, dilation_rate=dilation_rate)
    up22 = UpSampling2D(size=updown_sampling_size42, interpolation=upsample_interp)(bottleneck_2)

    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width * 4, use_batch_norm, dilation_rate=dilation_rate)

    dconv_up32 = double_conv(dconv_up3, width / 2, use_batch_norm, dilation_rate=dilation_rate)
    up32 = UpSampling2D(size=updown_sampling_size32, interpolation=upsample_interp)(dconv_up32)

    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, use_batch_norm, dilation_rate=dilation_rate)

    dconv_up22 = double_conv(dconv_up2, width / 2, use_batch_norm, dilation_rate=dilation_rate)
    up42 = UpSampling2D(size=updown_sampling_size22, interpolation=upsample_interp)(dconv_up22)

    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)

    dconv_up12 = double_conv(dconv_up1, width / 2, use_batch_norm, dilation_rate=dilation_rate)

    merge_n2 = concatenate([up22, up32, up42, dconv_up12], axis=3)
    dconv_n2 = double_conv(merge_n2, width, use_batch_norm, dilation_rate=dilation_rate)

    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_n2)

    return Model(inputs=inputs, outputs=conv_last)


def unethalf(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    updown_sampling_size_b = (8, 8)
    updown_sampling_size_d3 = (4, 4)
    updown_sampling_size_d2 = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled_2 = UpSampling2D(size=updown_sampling_size_d2, interpolation=upsample_interp)(dconv_down2)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled_3 = UpSampling2D(size=updown_sampling_size_d3, interpolation=upsample_interp)(dconv_down3)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled_b = UpSampling2D(size=updown_sampling_size_b, interpolation=upsample_interp)(bottleneck)
    summ = upsampled_2 + upsampled_3 + upsampled_b
    dconv_summ = double_conv(summ, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate,
                       kernel_initializer='he_normal')(dconv_summ)

    return Model(inputs=inputs, outputs=conv_last)


def bmunethalf(n_class, batch_size, input_size, width_div: int = 1, with_bg=False, use_batch_norm=True, dilation_rate=(1, 1), train=True):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    updown_sampling_size_b = (8, 8)
    updown_sampling_size_d3 = (4, 4)
    updown_sampling_size_d2 = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled_2 = UpSampling2D(size=updown_sampling_size_d2, interpolation=upsample_interp)(dconv_down2)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled_3 = UpSampling2D(size=updown_sampling_size_d3, interpolation=upsample_interp)(dconv_down3)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled_b = UpSampling2D(size=updown_sampling_size_b, interpolation=upsample_interp)(bottleneck)
    summ = upsampled_2 + upsampled_3 + upsampled_b
    dconv_summ = double_conv(summ, width, use_batch_norm, dilation_rate=dilation_rate)
    out_bin = Conv2D(2 if with_bg else 1, kernel_size=1, activation='sigmoid', padding='same',
                     dilation_rate=dilation_rate, name='out_bin')(dconv_summ)
    merge_img_bin_conv = concatenate([out_bin, dconv_summ], axis=3)
    conv_up1_1 = double_conv(merge_img_bin_conv, width // 2, use_batch_norm, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate,
                      name='out_mult')(conv_up1_1)

    if train:
        return Model(
            inputs=inputs,
            outputs={
                'out_bin': out_bin,
                'out_mult': out_mult
            }
        )
    else:
        return Model(inputs=inputs, outputs=out_mult)


def unetli(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)

    return Model(inputs=inputs, outputs=conv_last)


def bmunetli(n_class, batch_size, input_size, width_div: int, with_bg=False, use_batch_norm=False, dilation_rate=(1, 1), train=True):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, use_batch_norm, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    out_bin = Conv2D(2 if with_bg else 1, kernel_size=1, activation='sigmoid', padding='same',
                     dilation_rate=dilation_rate, name='out_bin')(dconv_up1)
    merge_img_bin_conv = concatenate([out_bin, dconv_up1], axis=3)
    conv_up1_1 = double_conv(merge_img_bin_conv, width // 2, use_batch_norm, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, name='out_mult')(conv_up1_1)

    if train:
        return Model(
            inputs=inputs,
            outputs={
                'out_bin': out_bin,
                'out_mult': out_mult
            }
        )
    else:
        return Model(inputs=inputs, outputs=out_mult)


def unetliplus(n_class, batch_size, input_size, width_div: int = 1, use_batch_norm=True, dilation_rate=(1, 1)):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = upsampled3 + dconv_down3
    dconv_up3 = double_conv(merge3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = upsampled2 + dconv_down2
    dconv_up2 = double_conv(merge2, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = upsampled1 + dconv_down1
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)

    return Model(inputs=inputs, outputs=conv_last)


def bmunetliplus(n_class, batch_size, input_size, width_div: int, with_bg=False, use_batch_norm=False, dilation_rate=(1, 1), train=True):
    width = 64 // width_div
    updown_sampling_size = (2, 2)
    if type(input_size) is tuple:
        input_shape = (input_size[0], input_size[1], 3)
    else:
        input_shape = (input_size, input_size, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, use_batch_norm, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width, use_batch_norm, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width, use_batch_norm, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = upsampled3 + dconv_down3
    dconv_up3 = double_conv(merge3, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = upsampled2 + dconv_down2
    dconv_up2 = double_conv(merge2, width, use_batch_norm, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = upsampled1 + dconv_down1
    dconv_up1 = double_conv(merge1, width, use_batch_norm, dilation_rate=dilation_rate)
    out_bin = Conv2D(2 if with_bg else 1, kernel_size=1, activation='sigmoid', padding='same',
                     dilation_rate=dilation_rate, name='out_bin')(dconv_up1)
    merge_img_bin_conv = concatenate([out_bin, dconv_up1], axis=3)
    conv_up1_1 = double_conv(merge_img_bin_conv, width // 2, use_batch_norm, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, name='out_mult')(conv_up1_1)

    if train:
        return Model(
            inputs=inputs,
            outputs={
                'out_bin': out_bin,
                'out_mult': out_mult
            }
        )
    else:
        return Model(inputs=inputs, outputs=out_mult)

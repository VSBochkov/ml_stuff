import json
import os
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from keras_dataset import ValidationDataset
from keras_unet_models import wuunet, bmunet, unet, uunet, unetpp, unetfpn, unethalf, unet2, unet3, unet5, unetli, \
    unetliplus, bmunethalf, bmunetli, bmunetliplus
from keras_deeplab_models import bmdeeplab, deeplab, bmdeeplabhalf, bmdeeplabli, deeplabhalf, deeplabli, deeplab5
from keras_segment_schemas import one_window, full_size_cut, full_size_lin, \
    half_intersected_add, half_intersected_addw, half_intersected_gauss
import keras_utils


class Validation(object):
    def __init__(self, dataset_variant: str, is_augm: bool = False, is_binary: bool = False,
                 is_half_intersect: bool = False, is_use_gauss: bool = False, is_weight: bool = False,
                 is_linear: bool = False, is_use_original_dataset: bool = False, is_save_mask: bool = False):
        if is_binary:
            self.mult_num_class = 1
        else:
            self.mult_num_class = 3
        self.is_augm = is_augm
        self.is_half_intersect = is_half_intersect
        self.is_use_gauss = is_use_gauss
        self.is_weight = is_weight
        self.is_linear = is_linear
        self.is_use_original_dataset = is_use_original_dataset
        self.is_save_mask = is_save_mask
        self.division_coeff = 1
        self.dataset_variant = dataset_variant
        self.sample_path_list = []
        self.ad_segmask_list = []
        self.gt_segmask_list = []
        self.schema_type = None
        self.win_size = None
        self.calculate = None

    @staticmethod
    def get_thresholds_array(begin, end, nodes):
        return np.linspace(begin, end, nodes)

    def get_model(self, model_name, snapshot_variant):
        if self.is_use_original_dataset or (self.schema_type == 'ow' and self.win_size == (360, 640)) or self.is_half_intersect:
            dataset = ValidationDataset('../dataset/original/' + self.dataset_variant, is_binary=self.mult_num_class == 1)
        else:
            dataset = ValidationDataset(f'../dataset/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/{self.dataset_variant}', is_binary=self.mult_num_class == 1)
        use_bn = 'bn' in model_name
        with_bg = 'bg' in model_name
        if '_n1' in model_name:
            n_class = 1
        elif with_bg:
            n_class = self.mult_num_class + 1
        else:
            n_class = self.mult_num_class
        l, wd = model_name.find('light'), model_name.find('wd')
        if l != -1:
            width_div = 2
        elif wd != -1:
            width_div = int(model_name[wd + 2])
        else:
            width_div = 1
        dil = model_name.find('dil')
        if dil != -1:
            dilation_rate = int(model_name[dil + 3])
        main_metric = 'jaccard'
        if 'wuunet' in model_name:
            print('wuunet')
            model = wuunet(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, use_batch_norm=use_bn, width_div=width_div, train=False)
        elif 'uunet' in model_name:
            print('uunet')
            model = uunet(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, use_batch_norm=use_bn, width_div=width_div, train=False)
        elif 'bmunethalf' in model_name:
            print('bmunethalf')
            model = bmunethalf(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, use_batch_norm=use_bn, width_div=width_div, train=False)
        elif 'bmunetliplus' in model_name:
            print('bmunetliplus')
            model = bmunetliplus(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, use_batch_norm=use_bn, width_div=width_div, train=False)
        elif 'bmunetli' in model_name:
            print('bmunetli')
            model = bmunetli(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, use_batch_norm=use_bn, width_div=width_div, train=False)
        elif 'bmunet' in model_name:
            print('bmunet')
            model = bmunet(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, use_batch_norm=use_bn, width_div=width_div, train=False)
        elif 'unet2' in model_name:
            print('unet2')
            model = unet2(n_class, batch_size=1, input_size=self.win_size, use_batch_norm=use_bn, width_div=width_div)
        elif 'unet3' in model_name:
            print('unet3')
            model = unet3(n_class, batch_size=1, input_size=self.win_size, use_batch_norm=use_bn, width_div=width_div)
        elif 'unet5' in model_name:
            print('unet5')
            model = unet5(n_class, batch_size=1, input_size=self.win_size, use_batch_norm=use_bn, width_div=width_div)
        elif 'unetpp_l3' in model_name:
            print('unetpp_l3')
            model = unetpp(n_class, batch_size=1, input_size=self.win_size, use_batch_norm=use_bn, out_aver_num=3)
        elif 'unetpp_l1' in model_name:
            print('unetpp_l1')
            model = unetpp(n_class, batch_size=1, input_size=self.win_size, use_batch_norm=use_bn, out_aver_num=1)
        elif 'unetpp_l2' in model_name:
            print('unetpp_l2')
            model = unetpp(n_class, batch_size=1, input_size=self.win_size, use_batch_norm=use_bn, out_aver_num=2)
        elif 'unetfpn' in model_name:
            print('unetfpn')
            model = unetfpn(n_class, batch_size=None, input_size=self.win_size, use_batch_norm=use_bn)
        elif 'unethalf' in model_name:
            print('unethalf')
            model = unethalf(n_class, batch_size=None, width_div=1, input_size=self.win_size, use_batch_norm=use_bn)
        elif 'unetliplus' in model_name:
            print('unetliplus')
            model = unetliplus(n_class, batch_size=None, width_div=1, input_size=self.win_size, use_batch_norm=use_bn)
        elif 'unetli' in model_name:
            print('unetli')
            model = unetli(n_class, batch_size=None, width_div=1, input_size=self.win_size, use_batch_norm=use_bn)
        elif 'unet' in model_name:
            print('unet')
            model = unet(n_class, batch_size=1, input_size=self.win_size, use_batch_norm=use_bn, width_div=width_div, dilation_rate=dilation_rate)
        elif 'bmdeeplabhalf' in model_name:
            print('bmdeeplabhalf')
            model = bmdeeplabhalf(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, width_div=width_div, train=False)
        elif 'bmdeeplabli' in model_name:
            print('bmdeeplabli')
            model = bmdeeplabli(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, width_div=width_div, train=False)
        elif 'bmdeeplab' in model_name:
            print('bmdeeplab')
            model = bmdeeplab(n_class, batch_size=1, input_size=self.win_size, with_bg=with_bg, width_div=width_div, train=False)
        elif 'deeplabhalf' in model_name:
            print('deeplabhalf')
            model = deeplabhalf(n_class, batch_size=1, input_size=self.win_size, width_div=width_div)
        elif 'deeplabli' in model_name:
            print('deeplabli')
            model = deeplabli(n_class, batch_size=1, input_size=self.win_size, width_div=width_div)
        elif 'deeplab5' in model_name:
            print('deeplab5')
            model = deeplab5(n_class, batch_size=1, input_size=self.win_size, width_div=width_div)
        else:
            print('deeplab')
            model = deeplab(n_class, batch_size=1, input_size=self.win_size, width_div=width_div)
        high_level_dir = 'output/keras'
        model.load_weights(join(high_level_dir, model_name, 'snapshots', snapshot_variant))
        return model, n_class, main_metric, dataset

    def init_calculate(self, ml_model_name):
        if 'fs' in ml_model_name:
            if self.is_half_intersect:
                if self.is_use_gauss:
                    self.calculate = half_intersected_gauss
                elif self.is_weight:
                    self.calculate = half_intersected_addw
                else:
                    self.calculate = half_intersected_add
            else:
                if self.is_linear:
                    self.calculate = full_size_lin
                else:
                    self.calculate = full_size_cut
        else:
            self.calculate = one_window

    def get_metrics(self, ad_mult, gt_mult):
        ad_bin = keras_utils.oh_mult2bin(ad_mult)
        gt_bin = keras_utils.oh_mult2bin(gt_mult)
        jaccard_mults = keras_utils.np_jaccard_acc(ad_mult, gt_mult)
        jaccard_bins = keras_utils.np_jaccard_acc(ad_bin, gt_bin)
        return jaccard_mults.mean(), jaccard_bins.mean()

    def get_metrics_for_all_samples(self, model_name, snapshot_variant, threshold, samples_num):
        model_metrics_dir = f'output/evaluation/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/{model_name}/{snapshot_variant}/'
        _jaccard_mult_list = np.zeros(samples_num, dtype=np.float32)
        _jaccard_bin_list = np.zeros(samples_num, dtype=np.float32)
        for j in range(0, samples_num):
            outputs = self.ad_segmask_list[j]
            mult_lbl = self.gt_segmask_list[j]
            outs = outputs - threshold
            outs = outs.clip(min=0)
            maximals = outs.max(axis=2, keepdims=True)
            argmaximals = np.expand_dims(outs.argmax(axis=2), axis=2)
            outs = argmaximals + np.asarray(maximals > 0, dtype=np.int64)
            outs = np.asarray(np.squeeze(outs), dtype=np.uint8)
            if self.is_save_mask:
                video, sample = self.sample_path_list[j]
                video_dir = model_metrics_dir + f'{video}'
                os.makedirs(video_dir, exist_ok=True)
                np.save(f'{video_dir}/{sample}.npy', outs)
            outs = keras_utils.lbl2oh_mult(outs, self.mult_num_class)
            _jaccard_mult_list[j], _jaccard_bin_list[j] = self.get_metrics(outs, mult_lbl)
        return _jaccard_mult_list, _jaccard_bin_list

    def metrics(self, samples_num, threshold, model_name, snapshot_variant):
        _jaccard_mult_list, _jaccard_bin_list = self.get_metrics_for_all_samples(
            model_name,
            snapshot_variant,
            threshold,
            samples_num
        )
        jaccard_mult_mean = _jaccard_mult_list.mean()
        jaccard_mult_std = np.std(_jaccard_mult_list)
        jaccard_bin_mean = _jaccard_bin_list.mean()
        jaccard_bin_std = np.std(_jaccard_bin_list)
        model_metrics_dir = f'output/evaluation/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/{model_name}/{snapshot_variant}/'
        if self.is_save_mask:
            samples = np.asarray(self.sample_path_list, dtype=np.uint8)
            np.save(f'{model_metrics_dir}/jaccard_mult.npy', _jaccard_mult_list)
            np.save(f'{model_metrics_dir}/jaccard_bin.npy', _jaccard_bin_list)
            np.save(f'output/evaluation/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/samples.npy', samples)
        return jaccard_mult_mean, jaccard_mult_std, jaccard_bin_mean, jaccard_bin_std

    def get_thresholded_metrics(self, activation_thrs, model_name, snapshot_variant, samples_num, visualize: bool = False):
        jaccard_mult_mean = np.zeros(len(activation_thrs), dtype=np.float32)
        jaccard_mult_std = np.zeros(len(activation_thrs), dtype=np.float32)
        jaccard_bin_mean = np.zeros(len(activation_thrs), dtype=np.float32)
        jaccard_bin_std = np.zeros(len(activation_thrs), dtype=np.float32)
        for i, threshold in enumerate(tqdm(activation_thrs)):
            jaccard_mult_mean[i], jaccard_mult_std[i], jaccard_bin_mean[i], jaccard_bin_std[i] = self.metrics(
                samples_num, threshold, model_name, snapshot_variant)
        if visualize:
            self.draw_threshold_graphics(jaccard_mult_mean, activation_thrs)
        best_thr_id = np.argmax(jaccard_mult_mean)
        best_thr = activation_thrs[best_thr_id]
        return best_thr, jaccard_mult_mean[best_thr_id], jaccard_mult_std[best_thr_id], jaccard_bin_mean[best_thr_id], jaccard_bin_std[best_thr_id]

    def get_all_metrics(self, model, dataset):
        self.ad_segmask_list = []
        self.gt_segmask_list = []
        for i in range(0, len(dataset)):
            image, label, video_num, sample_num = dataset[i]
            # print('image.shape', image.shape, 'dtype', image.dtype, image.max(), 'label.shape', label.shape)
            output_mul = self.calculate(image, model, self.win_size)
            ad_segmask = output_mul.astype(np.float32)
            gt_segmask = label.astype(np.float32)
            self.ad_segmask_list.append(ad_segmask)  # -1,+1 but not binary
            self.gt_segmask_list.append(gt_segmask)
            self.sample_path_list.append((video_num, sample_num))
        return len(dataset)

    def autolabel(self, ax, rects, width):
        x = rects[0].get_x() - width / 10
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height > 0.:
                ax.text(x, height, '{0:.4f}'.format(height), ha='right', va='bottom')

    def draw_barchart(self, model_metrics, step=4.5):
        jaccard_mult_means = [0]
        jaccard_mult_means.extend([model_metrics[model_name][3] for model_name in model_metrics])
        jaccard_mult_stds = [0]
        jaccard_mult_stds.extend([model_metrics[model_name][4] for model_name in model_metrics])
        jaccard_bin_means = [0]
        jaccard_bin_means.extend([model_metrics[model_name][7] for model_name in model_metrics])
        jaccard_bin_stds = [0]
        jaccard_bin_stds.extend([model_metrics[model_name][8] for model_name in model_metrics])

        ind = np.arange(len(model_metrics) + 1) * step  # the x locations for the groups
        width = step / 6  # the width of the bars

        fig, ax = plt.subplots(figsize=(len(model_metrics) * 4 + 6, 8))
        colors = ['#7f7f7f', '#9f9f9f', '#bfbfbf', '#dfdfdf']
        rects3 = ax.bar(ind + 2 * width + 1, jaccard_mult_means, width, yerr=jaccard_mult_stds,
                        color=colors[3], label='JACCARD MULTICLASS')

        for i in range(0, len(rects3)):
            self.autolabel(ax, [rects3[i]], width)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Value')
        ax.set_title('Pixel-wise segmentation metrics')
        ax.set_xticks(ind + step)
        labels = ['']
        labels.extend(model_metrics.keys())
        ax.set_xticklabels(labels)
        ax.legend(loc='upper left')

        plt.show()

    def draw_max_metric(self, plt, metric_id, metric, metric_name, ytext):
        plt.annotate(metric_name + ' = {0:.3f}'.format(metric), xy=(self.activation_thrs[metric_id], metric), xytext=(self.activation_thrs[metric_id] + 0.01, ytext))#,
        plt.axvline(x=self.activation_thrs[metric_id], ymin=0, ymax=ytext, linestyle='--', color='#7f7f7f')

    def plot_thresholds(self, jaccard_mults, jaccard_bins):
        plt.figure(figsize=(15, 5))
        plt.plot(self.activation_thrs, jaccard_bins, linestyle='--', color='#7f7f7f')
        plt.plot(self.activation_thrs, jaccard_mults, linestyle='--', color='#afafaf')
        legend = ['BINARY JACCARD', 'MULTICLASS JACCARD']
        plt.grid(True)
        plt.legend(legend, loc='upper left', fontsize=14)
        plt.xlabel('threshold (distance)', fontsize=18)
        plt.ylabel('rate', fontsize=18)
        jba = np.argmax(jaccard_bins)
        jma = np.argmax(jaccard_mults)
        metrics = np.array([jaccard_mults[jma]])
        self.draw_max_metric(plt, jma, jaccard_mults[jma], 'MULT JACCARD MAX', metrics.max() + 0.25)
        self.draw_max_metric(plt, jba, jaccard_bins[jba], 'BIN JACCARD MAX', metrics.max() + 0.15)
        plt.ylim(top=metrics.max() + 0.30)
        plt.tick_params(labelsize=14)

    def parse_img_params(self, ml_model_name: str):
        if 'fs' in ml_model_name:
            self.schema_type = 'fs'
            if self.is_half_intersect:
                if self.is_use_gauss:
                    self.schema_type += '_gauss'
                elif self.is_weight:
                    self.schema_type += '_addw'
                else:
                    self.schema_type += '_addw'
            # else:
            #     if self.is_linear:
            #         self.schema_type += '_lin'
            #     else:
            #         self.schema_type += '_cut'
        elif 'ow' in ml_model_name:
            self.schema_type = 'ow'
        if '224' in ml_model_name:
            self.win_size = 224
        elif '304' in ml_model_name:
            self.win_size = 304
        elif '336' in ml_model_name:
            self.win_size = 336
        elif '360' in ml_model_name:
            self.win_size = (360, 640) if 'ow' in ml_model_name else 360
            # self.schema_type = None

    def calculate_thresholds(self, ml_model_name, snapshot_variant, visualize: bool = False):
        self.parse_img_params(ml_model_name)
        self.init_calculate(ml_model_name)
        print(self.schema_type, self.win_size)
        model_metrics_dir = f'output/evaluation/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/{ml_model_name}/{snapshot_variant}/'
        print("model_metrics_dir", model_metrics_dir)
        if os.path.exists(join(model_metrics_dir, 'acc.npy')):
            acc = np.load(join(model_metrics_dir, 'acc.npy'))
            best_thr, jm_mean, jm_std, jb_mean, jb_std = acc[0], acc[1], acc[2], acc[3], acc[4]
        else:
            best_thr, jm_mean, jm_std, jb_mean, jb_std = self.find_best_threshold(ml_model_name, snapshot_variant, visualize)
            if not os.path.exists(model_metrics_dir):
                os.makedirs(model_metrics_dir)
            acc = np.asarray([best_thr, jm_mean, jm_std, jb_mean, jb_std])
            np.save(join(model_metrics_dir, 'acc.npy'), acc)
        print('[{}_{}, threshold = {}]:\n'
              'MULTICLASS JACCARD [val = {}, std = {}]\n'
              'BINARY JACCARD [val = {}, std = {}]\n'
                  .format(ml_model_name, snapshot_variant, best_thr, jm_mean, jm_std, jb_mean, jb_std))
        #self.plot_thresholds(dm_mean, jm_mean, db_mean, jb_mean)
        return best_thr, jm_mean, jm_std, jb_mean, jb_std

    def find_best_threshold(self, ml_model_name: str, snapshot_variant: str, visualize: bool = False):
        model, num_classes, main_metric, dl_test = self.get_model(ml_model_name, snapshot_variant)
        samples_num = self.get_all_metrics(model, dl_test)
        best_thr, jm_mean, jm_std, jb_mean, jb_std = 0., 0., 0., 0., 0.
        # begin, end, step, nodes = 0.0, 1.0, 0.1, 11
        begin, end, step, nodes = 0.1, 0.9, 0.1, 9
        for i in range(0, 5):
            activation_thrs = self.get_thresholds_array(begin=begin, end=end, nodes=nodes)
            best_thr, jm_mean, jm_std, jb_mean, jb_std = self.get_thresholded_metrics(
                activation_thrs, ml_model_name, snapshot_variant, samples_num, visualize)
            # begin, end, step, nodes = best_thr - step, best_thr + step, step / 10, 21
            begin, end, step, nodes = best_thr - step * 0.9, best_thr + step * 0.9, step / 10, 19
        return best_thr, jm_mean, jm_std, jb_mean, jb_std

    def calculate_with_thr(self, ml_model_name: str, snapshot_variant: str, threshold: float):
        self.parse_img_params(ml_model_name)
        self.init_calculate(ml_model_name)
        print(self.schema_type, self.win_size)
        model, num_classes, main_metric, dl_test = self.get_model(ml_model_name, snapshot_variant)
        samples_num = self.get_all_metrics(model, dl_test)
        jmm, jms, jbm, jbs = self.metrics(samples_num, threshold, ml_model_name, snapshot_variant)
        print('[{}_{}, threshold = {}]:\n'
              'MULTICLASS JACCARD [val = {}, std = {}]\n'
              'BINARY JACCARD [val = {}, std = {}]\n'
                  .format(ml_model_name, snapshot_variant, threshold, jmm, jms, jbm, jbs))


    def print_metrics_at(self, ml_model_name, snapshot_variant, thr):
        model_metrics_dir = f'output/evaluation/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/{ml_model_name}/{snapshot_variant}/'
        if os.path.exists(model_metrics_dir):
            jm_mean = np.load(join(model_metrics_dir, 'jaccard_mult_mean.npy'))
            jm_std = np.load(join(model_metrics_dir, 'jaccard_mult_std.npy'))
            jb_mean = np.load(join(model_metrics_dir, 'jaccard_bin_mean.npy'))
            jb_std = np.load(join(model_metrics_dir, 'jaccard_bin_std.npy'))
            thr_id = self.activation_thrs.index(thr)
            print('[{}_{}, threshold = {}]:\n'
                  'MULTICLASS JACCARD [val = {}, std = {}]\n'
                  'BINARY JACCARD [val = {}, std = {}]\n'
                  .format(ml_model_name, snapshot_variant, thr, jm_mean[thr_id], jm_std[thr_id], jb_mean[thr_id], jb_std[thr_id]))
    #
    # ##############################
    # ######## VIZUALIZATION #######
    # ##############################

    def draw_threshold_graphics(self, thr_accs: np.array, thresholds: np.array):
        fig = plt.figure()
        fig.add_subplot(111)
        X, Y = thresholds, thr_accs
        print(f'X = {X}, Y = {Y}')
        amax_thr_acc = np.argmax(thr_accs)
        max_thr_acc = thr_accs[amax_thr_acc]
        best_thr = thresholds[amax_thr_acc]
        plt.plot(X, Y, 'b-')
        plt.plot(best_thr, max_thr_acc, 'bo')
        # plt.yscale('log')
        plt.grid(True)
        plt.axvline(x=thresholds[amax_thr_acc - 1], linestyle='--', color='b')
        plt.axvline(x=thresholds[amax_thr_acc + 1], linestyle='--', color='b')
        fig.canvas.draw()
        plt.show()
        # # Now we can save it to a numpy array.
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # return data


class ValidationWithBG(Validation):
    def __init__(self, dataset_variant: str, is_half_intersect: bool):
        super().__init__(dataset_variant=dataset_variant, is_half_intersect=is_half_intersect)

    def get_jaccard_acc(self, ad_mult, gt_mult):
        ad_bin = keras_utils.oh_mult2bin_with_bg(ad_mult)
        gt_bin = keras_utils.oh_mult2bin_with_bg(gt_mult)
        jaccard_mults = keras_utils.np_jaccard_acc(ad_mult, gt_mult)
        jaccard_bins = keras_utils.np_jaccard_acc(ad_bin, gt_bin)
        return jaccard_mults.mean(), jaccard_bins.mean()

    def get_metrics(self, model, dataset):
        jaccard_mult = np.zeros(len(dataset), dtype=np.float32)
        jaccard_bin = np.zeros(len(dataset), dtype=np.float32)
        for i in range(0, len(dataset)):
            image, gt_one_hot, video_num, sample_num = dataset[i]
            gt_one_hot = keras_utils.oh2oh_with_bg_mult(gt_one_hot)
            output_mul = self.calculate(image, model, self.win_size).astype(np.float32)
            ad_mult_labeled = np.expand_dims(output_mul.argmax(axis=2), axis=2)
            ad_one_hot = keras_utils.lbl2oh_mult_with_bg(np.squeeze(ad_mult_labeled), self.mult_num_class)
            jaccard_mult[i], jaccard_bin[i] = self.get_jaccard_acc(ad_one_hot, gt_one_hot)
            self.sample_path_list.append((video_num, sample_num))
        jm_mean, jm_std = jaccard_mult.mean(), np.std(jaccard_mult)
        jb_mean, jb_std = jaccard_bin.mean(), np.std(jaccard_bin)
        return jm_mean, jm_std, jb_mean, jb_std

    def evaluate(self, ml_model_name: str, snapshot_variant: str):
        self.parse_img_params(ml_model_name)
        print(self.schema_type, self.win_size)
        self.init_calculate(ml_model_name)
        model, n_class, main_metric, dataset = self.get_model(ml_model_name, snapshot_variant)
        jm_mean, jm_std, jb_mean, jb_std = self.get_metrics(model, dataset)
        print('[{}_{}]:\n'
              'MULTICLASS JACCARD [val = {}, std = {}]\n'
              'BINARY JACCARD [val = {}, std = {}]\n'
              .format(ml_model_name, snapshot_variant, jm_mean, jm_std, jb_mean, jb_std))
        return jm_mean, jm_std, jb_mean, jb_std


class ValidationWithBG_AsNotBG(Validation):
    def __init__(self, dataset_variant: str, is_augm: bool = False, is_binary: bool = False,
                 is_half_intersect: bool = False, is_use_gauss: bool = False, is_weight: bool = False,
                 is_linear: bool = False, is_use_original_dataset: bool = False, is_save_mask: bool = False):
        super().__init__(dataset_variant=dataset_variant, is_augm=is_augm, is_binary=is_binary,
                         is_half_intersect=is_half_intersect, is_use_gauss=is_use_gauss, is_weight=is_weight,
                         is_linear=is_linear, is_use_original_dataset=is_use_original_dataset, is_save_mask=is_save_mask)

    def get_jaccard_acc(self, ad_mult, gt_mult):
        ad_bin = keras_utils.oh_mult2bin(ad_mult)
        gt_bin = keras_utils.oh_mult2bin(gt_mult)
        jaccard_mults = keras_utils.np_jaccard_acc(ad_mult, gt_mult)
        jaccard_bins = keras_utils.np_jaccard_acc(ad_bin, gt_bin)
        return jaccard_mults.mean(), jaccard_bins.mean()

    def get_metrics(self, model, dataset, ml_model_name, snapshot_variant):
        model_metrics_dir = f'output/evaluation/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/{ml_model_name}/{snapshot_variant}/'
        jaccard_mult = np.zeros(len(dataset), dtype=np.float32)
        jaccard_bin = np.zeros(len(dataset), dtype=np.float32)
        for i in range(0, len(dataset)):
            image, gt_one_hot, video_num, sample_num = dataset[i]
            to_interpolate = self.is_use_original_dataset and image.shape[0] != self.win_size
            if to_interpolate:
                image = cv2.resize(image, (self.win_size, self.win_size), interpolation=cv2.INTER_CUBIC)
            output_mul = self.calculate(image, model, self.win_size).astype(np.float32)
            if to_interpolate:
                output_mul = cv2.resize(output_mul, (640, 360), interpolation=cv2.INTER_CUBIC)
            ad_mult_labeled = np.expand_dims(output_mul.argmax(axis=2), axis=2)
            ad_mult_labeled = np.asarray(np.squeeze(ad_mult_labeled), dtype=np.uint8)
            if self.is_save_mask:
                video_dir = model_metrics_dir + f'{video_num}'
                os.makedirs(video_dir, exist_ok=True)
                np.save(f'{video_dir}/{sample_num}.npy', ad_mult_labeled)
                self.sample_path_list.append((video_num, sample_num))
            ad_one_hot = keras_utils.lbl2oh_mult(ad_mult_labeled, self.mult_num_class)
            jaccard_mult[i], jaccard_bin[i] = self.get_jaccard_acc(ad_one_hot, gt_one_hot)
        jm_mean, jm_std = jaccard_mult.mean(), np.std(jaccard_mult)
        jb_mean, jb_std = jaccard_bin.mean(), np.std(jaccard_bin)
        if self.is_save_mask:
            np.save(f'{model_metrics_dir}/jaccard_mult.npy', jaccard_mult)
            np.save(f'{model_metrics_dir}/jaccard_bin.npy', jaccard_bin)
        return jm_mean, jm_std, jb_mean, jb_std

    def evaluate(self, ml_model_name: str, snapshot_variant: str):
        self.parse_img_params(ml_model_name)
        print(self.schema_type, self.win_size)
        self.init_calculate(ml_model_name)
        model, n_class, main_metric, dataset = self.get_model(ml_model_name, snapshot_variant)
        jm_mean, jm_std, jb_mean, jb_std = self.get_metrics(model, dataset, ml_model_name, snapshot_variant)
        if self.is_save_mask:
            samples = np.asarray(self.sample_path_list, dtype=np.uint8)
            np.save(f'output/evaluation/{self.schema_type}_{self.win_size}{"_augm" if self.is_augm else ""}/samples.npy', samples)
        print('[{}_{}]:\n'
              'MULTICLASS JACCARD [val = {}, std = {}]\n'
              'BINARY JACCARD [val = {}, std = {}]\n'
              .format(ml_model_name, snapshot_variant, jm_mean, jm_std, jb_mean, jb_std))
        return jm_mean, jm_std, jb_mean, jb_std


class BinValidationWithBG_AsNotBG(Validation):
    def __init__(self, dataset_variant: str):
        super().__init__(dataset_variant=dataset_variant, is_binary=True)

    def get_jaccard_acc(self, ad, gt):
        jaccard = keras_utils.np_jaccard_acc(np.expand_dims(ad, axis=2), np.expand_dims(gt, axis=2))
        return jaccard.mean()

    def get_metrics(self, model, dataset):
        jaccard = np.zeros(len(dataset), dtype=np.float32)
        for i in range(0, len(dataset)):
            image, gt_one_hot = dataset[i]
            output = self.calculate(image, model, self.win_size).astype(np.float32)
            ad_labeled = output.argmax(axis=2)
            gt_labeled = keras_utils.oh_with_bg2lbl_mult(gt_one_hot)
            jaccard[i] = self.get_jaccard_acc(ad_labeled, gt_labeled)
        j_mean, j_std = jaccard.mean(), np.std(jaccard)
        return j_mean, j_std

    def evaluate(self, ml_model_name: str, snapshot_variant: str):
        self.parse_img_params(ml_model_name)
        print(self.schema_type, self.win_size)
        self.init_calculate(ml_model_name)
        model, n_class, main_metric, dataset = self.get_model(ml_model_name, snapshot_variant)
        j_mean, j_std = self.get_metrics(model, dataset)
        print('[{}_{}]:\n'
              'BINARY JACCARD [val = {}, std = {}]\n'
              .format(ml_model_name, snapshot_variant, j_mean, j_std))
        return j_mean, j_std


if __name__ == '__main__':
    validation = Validation('test')
    validation.calculate_thresholds('unet_bn_light_224_ow', 'best_val_loss_1913.hdf5', visualize=True)
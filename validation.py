import json
import os
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

from color_map.color_map_segm_model import ColorMapModel
from ml_models.one_shot_knn import OneShotKNN
from segmentation_model import SegmentationModel

import utils


class Validation(object):
    def __init__(self, dataset_path: str, segmentation_model: SegmentationModel):
        self.dataset_path = dataset_path
        self.segmentation_model = segmentation_model

    def draw_mask(self, image: np.array, oh_label: np.array, overlay: np.array,
                  red_color: np.array, orange_color: np.array, yellow_color: np.array):
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if oh_label[i, j, 0] > 0.1:
                    overlay[i, j] = red_color
                elif oh_label[i, j, 1] > 0.1:
                    overlay[i, j] = orange_color
                elif oh_label[i, j, 2] > 0.1:
                    overlay[i, j] = yellow_color
        return overlay

    def get_overlay(self, image: np.array, oh_label: np.array):
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        red_color = np.asarray([0, 0, 0xff], dtype=np.uint8)
        orange_color = np.asarray([0, 0xcc, 0xff], dtype=np.uint8)
        yellow_color = np.asarray([0, 0xff, 0xff], dtype=np.uint8)
        return self.draw_mask(image, oh_label, overlay, red_color, orange_color, yellow_color)

    def get_mask(self, image: np.array, oh_label: np.array):
        mask_img = np.ones_like(image) * 0xff
        red_color = np.asarray([0, 0, 0xff], dtype=np.uint8)
        orange_color = np.asarray([0, 0xcc, 0xff], dtype=np.uint8)
        yellow_color = np.asarray([0, 0xff, 0xff], dtype=np.uint8)
        return self.draw_mask(image, oh_label, mask_img, red_color, orange_color, yellow_color)

    def get_metrics(self):
        metrics_dict = {}
        return metrics_dict


class BinarizedModelValidation(Validation):
    def __init__(self, dataset_path: str, segmentation_model: SegmentationModel):
        super().__init__(dataset_path, segmentation_model)

    def get_metrics(self):
        images_dir = join(self.dataset_path, 'images')
        labels_dir = join(self.dataset_path, 'labels')
        overlay_dir = join(self.dataset_path, self.segmentation_model.name)
        jaccard_accs = []
        bin_jaccard_accs = []
        bin_miss_rates = []
        red_miss_rates = []
        orange_miss_rates = []
        yellow_miss_rates = []
        for video_num in tqdm(os.listdir(images_dir)):
            for sample_num in range(0, len(os.listdir(join(images_dir, video_num)))):
                if os.path.isdir(join(images_dir, video_num, f'{sample_num}')):
                    anim = []
                    for img_num in range(0, len(os.listdir(join(images_dir, video_num, f'{sample_num}')))):
                        anim.append(cv2.imread(join(images_dir, video_num, f'{sample_num}', f'{img_num}.jpg')))
                    input = np.array(anim)
                    image = input[-1]
                else:
                    input = cv2.imread(join(images_dir, video_num, f'{sample_num}.jpg'))
                    image = input
                gt_label = np.load(join(labels_dir, video_num, f'{sample_num}.npy'))
                ad_oh = self.segmentation_model.predict(input)
                ad_label = utils.oh2lbl_mult(ad_oh)
                ad_red_oh = np.expand_dims(ad_oh[:, :, 0], axis=2)
                ad_orange_oh = np.expand_dims(ad_oh[:, :, 1], axis=2)
                ad_yellow_oh = np.expand_dims(ad_oh[:, :, 2], axis=2)
                mask = self.get_mask(image, ad_oh)
                overlay = self.get_overlay(image, ad_oh)
                os.makedirs(join(overlay_dir, video_num), exist_ok=True)
                cv2.imwrite(join(overlay_dir, video_num, f'{sample_num}_overlay.jpg'), overlay)
                cv2.imwrite(join(overlay_dir, video_num, f'{sample_num}_mask.jpg'), mask)
                ad_bin_oh = utils.oh_mult2bin(ad_oh)
                ad_bin_label = utils.oh2lbl_mult(ad_bin_oh)
                gt_oh = utils.lbl2oh_mult(gt_label, 3)
                gt_bin_oh = utils.oh_mult2bin(gt_oh)
                gt_bin_label = utils.oh2lbl_mult(gt_bin_oh)
                gt_red_oh = np.expand_dims(gt_oh[:, :, 0], axis=2)
                gt_orange_oh = np.expand_dims(gt_oh[:, :, 1], axis=2)
                gt_yellow_oh = np.expand_dims(gt_oh[:, :, 2], axis=2)
                jaccard_acc = utils.jaccard_acc(pred=ad_oh, target=gt_oh).mean()
                bin_jaccard_acc = utils.jaccard_acc(pred=ad_bin_oh, target=gt_bin_oh).mean()
                red_jaccard_acc = utils.jaccard_acc(pred=ad_red_oh, target=gt_red_oh).mean()
                orange_jaccard_acc = utils.jaccard_acc(pred=ad_orange_oh, target=gt_orange_oh).mean()
                yellow_jaccard_acc = utils.jaccard_acc(pred=ad_yellow_oh, target=gt_yellow_oh).mean()
                bin_confusion = utils.confusion(ad_bin_label, gt_bin_label, 2)
                bin_miss_rate = utils.miss_rate(bin_confusion, class_num=1)
                mult_confusion = utils.confusion(ad_label, gt_label, 4)
                red_miss_rate = utils.miss_rate(mult_confusion, class_num=1)
                orange_miss_rate = utils.miss_rate(mult_confusion, class_num=2)
                yellow_miss_rate = utils.miss_rate(mult_confusion, class_num=3)
                np.save(join(overlay_dir, video_num, f'{sample_num}_mult_confusion.npy'), mult_confusion)
                np.save(join(overlay_dir, video_num, f'{sample_num}_bin_confusion.npy'), bin_confusion)
                json.dump({
                    'iou_mult': float(jaccard_acc),
                    'iou_bin': float(bin_jaccard_acc),
                    'iou_red': float(red_jaccard_acc),
                    'iou_orange': float(orange_jaccard_acc),
                    'iou_yellow': float(yellow_jaccard_acc),
                    'miss_rate_bin': float(bin_miss_rate),
                    'miss_rate_red': float(red_miss_rate),
                    'miss_rate_orange': float(orange_miss_rate),
                    'miss_rate_yellow': float(yellow_miss_rate)
                }, open(join(overlay_dir, video_num, f'{sample_num}_acc.json'), mode='w'), indent=4)
                jaccard_accs.append(jaccard_acc)
                bin_jaccard_accs.append(bin_jaccard_acc)
                bin_miss_rates.append(bin_miss_rate)
                red_miss_rates.append(red_miss_rate)
                orange_miss_rates.append(orange_miss_rate)
                yellow_miss_rates.append(yellow_miss_rate)
        jaccard_accs = np.asarray(jaccard_accs)
        bin_jaccard_accs = np.asarray(bin_jaccard_accs)
        bin_miss_rates = np.asarray(bin_miss_rates)
        red_miss_rates = np.asarray(red_miss_rates)
        orange_miss_rates = np.asarray(orange_miss_rates)
        yellow_miss_rates = np.asarray(yellow_miss_rates)
        return {
            'Mult IOU mean': jaccard_accs.mean(),
            'Mult IOU std': jaccard_accs.std(),
            'Bin IOU mean': bin_jaccard_accs.mean(),
            'Bin IOU std': bin_jaccard_accs.std(),
            'Bin miss-rate mean': bin_miss_rates.mean(),
            'Bin miss-rate std': bin_miss_rates.std(),
            'Red miss-rate mean': red_miss_rates.mean(),
            'Red miss-rate std': red_miss_rates.std(),
            'Orange miss-rate mean': orange_miss_rates.mean(),
            'Orange miss-rate std': orange_miss_rates.std(),
            'Yellow miss-rate mean': yellow_miss_rates.mean(),
            'Yellow miss-rate std': yellow_miss_rates.std()
        }


if __name__ == '__main__':
    print(' Variant    | Dataset | ColorMapModel')
    print()
    # print('   train    | 224x224 | 224x224', BinarizedModelValidation(
    #     dataset_path='datasets/224x224/train',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/224x224/color_map.npy')
    # ).get_metrics())
    # print('    test    | 224x224 | 224x224', BinarizedModelValidation(
    #     dataset_path='datasets/224x224/test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/224x224/color_map.npy')
    # ).get_metrics())
    # print(' train_test | 224x224 | 224x224', BinarizedModelValidation(
    #     dataset_path='datasets/224x224/train_test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/224x224/color_map.npy')
    # ).get_metrics())
    # print()
    # print('   train    | 224x224 | 640x360', BinarizedModelValidation(
    #     dataset_path='datasets/224x224/train',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/640x360/color_map.npy')
    # ).get_metrics())
    # print('    test    | 224x224 | 640x360', BinarizedModelValidation(
    #     dataset_path='datasets/224x224/test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/640x360/color_map.npy')
    # ).get_metrics())
    # print(' train_test | 224x224 | 640x360', BinarizedModelValidation(
    #     dataset_path='datasets/224x224/train_test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/640x360/color_map.npy')
    # ).get_metrics())
    # print()
    # print('    train   | 640x360 | 224x224', BinarizedModelValidation(
    #     dataset_path='datasets/640x360/train',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/224x224/color_map.npy')
    # ).get_metrics())
    # print('    test    | 640x360 | 224x224', BinarizedModelValidation(
    #     dataset_path='datasets/640x360/test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/224x224/color_map.npy')
    # ).get_metrics())
    # print(' train_test | 640x360 | 224x224', BinarizedModelValidation(
    #     dataset_path='datasets/640x360/train_test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/224x224/color_map.npy')
    # ).get_metrics())
    # print()
    # print('  train    | 640x360 | 640x360', BinarizedModelValidation(
    #     dataset_path='datasets/640x360/train',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/640x360/color_map.npy')
    # ).get_metrics())
    # print('test       | 640x360 | 640x360', BinarizedModelValidation(
    #     dataset_path='datasets/640x360/test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/640x360/color_map.npy')
    # ).get_metrics())
    # print('train_test | 640x360 | 640x360', BinarizedModelValidation(
    #     dataset_path='datasets/640x360/train_test',
    #     segmentation_model=ColorMapModel('fire_color_map', 'color_map/640x360/color_map.npy')
    # ).get_metrics())

    train_X = np.load(join('datasets', '224x224_anim10', 'train', 'features.npy'))
    train_Y = np.load(join('datasets', '224x224_anim10', 'train', 'output.npy'))
    print('test | 224x224_anim10', BinarizedModelValidation(
        dataset_path=f'datasets/224x224_anim10/test',
        segmentation_model=OneShotKNN(
            f'one-shot_knn_224x224[3_uniform]',
            num_neighb=3,
            weights_type='uniform',
            train_x=train_X,
            train_y=train_Y
        )
    ).get_metrics())

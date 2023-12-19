import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sp
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
import cv2
import os
import glob
from tqdm import tqdm
import random

mask_dir = '/home/sdd/zxy/TCGA_data/blca_whole_slide_single_attn_visualization/train/head_max'
level_dir = 'F:\P__Gene_data\LUSC\level'
save_dir = '/home/sdd/zxy/TCGA_data/blca_whole_slide_single_attn_visualization/train/smoothed_head_max'

def run_smooth_img():
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_path_list = glob.glob(os.path.join(mask_dir, '*.npy'))
    random.shuffle(file_path_list)
    for file_path in tqdm(file_path_list):
        if os.path.exists(os.path.join(save_dir, os.path.basename(file_path).replace('.npy', '.png'))):
            continue
        # test_img = cv2.imread(r'F:\P__Gene_data\LUSC\attn_vis\head_max_12_12_0.1\TCGA-18-3406-01Z-00-DX1_0.png', 0)
        # test_img = plt.imread(r'F:\P__Gene_data\LUSC\attn_vis\head_max_12_12_0.1\TCGA-18-3406-01Z-00-DX1_0.png')
        test_img = np.load(file_path)
        test_img = cv2.resize(test_img, (test_img.shape[1] * 4, test_img.shape[0] * 4))
        smooth_img = convolve(test_img, Gaussian2DKernel(x_stddev=2))
        smooth_img =cv2.applyColorMap(np.uint8(255*smooth_img), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(file_path).replace('.npy', '.png')), smooth_img)


def run_add_smooth_img():
    level_image_list = glob.glob(os.path.join(level_dir, '*.jpg'))
    for image_path in tqdm(level_image_list):
        img_name = os.path.basename(image_path)[:os.path.basename(image_path).find('.')]
        mask_path_list = glob.glob(os.path.join(mask_dir, img_name + '*.npy'))
        if len(mask_path_list) > 0:
            file_path_list = glob.glob(os.path.join(mask_dir, '*.npy'))
            for file_path in tqdm(file_path_list):
                if os.path.exists(os.path.join(save_dir, os.path.basename(file_path).replace('.npy', '.png'))):
                    continue
                test_img = np.load(file_path)
                smooth_img = convolve(test_img, Gaussian2DKernel(x_stddev=2))
                smooth_img =cv2.applyColorMap(np.uint8(255*smooth_img), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(save_dir, os.path.basename(file_path).replace('.npy', '.png')), smooth_img)


if __name__ == '__main__':
    run_smooth_img()

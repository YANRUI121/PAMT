import os
import glob
import shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

image_dir = '/home/sda/zxy/luad_whole_slide_single_attn_visualization/train/gray_npy'
save_dir = '/home/sda/zxy/luad_whole_slide_single_attn_visualization/train/head_max'

if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_list = glob.glob(os.path.join(os.path.join(image_dir, 'head0'), '*.npy'))

    for image_path in tqdm(image_list):
        img_arr_list = []
        img_arr_list.append(np.load(image_path))
        for i in range(1, 16):
            temp_img_path = os.path.join(os.path.join(image_dir, 'head{}'.format(str(i))), os.path.basename(image_path))
            img_arr_list.append(np.load(temp_img_path))
            # 将图像数据堆叠成一个三维数组
        stacked_images = np.stack(img_arr_list)
        print(stacked_images.shape)
        # 计算每个位置的像素最大值
        max_pixels = np.max(stacked_images, axis=0)
        np.save(os.path.join(save_dir, os.path.basename(image_path)), max_pixels)
        res_map = cv2.applyColorMap(np.uint8(255*max_pixels), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path).replace('.npy','.png')), res_map)

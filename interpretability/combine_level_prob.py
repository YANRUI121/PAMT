import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import glob
from re import T
import shutil
import numpy as np
import torch
from tqdm import tqdm
import random
import cv2

#image_dir = '/home/sdc/tcga_sdd/luad_level1'
#mask_dir = '/home/sdd/zxy/TCGA_data/luad_whole_slide_single_attn_visualization/valid_80_80_80/head_max'
#save_dir = '/home/sda/tcga_attn_combine/luad'
image_dir = '/home/sdc/tcga_sdd/blca/blca_level1'
mask_dir = '/home/sdd/zxy/TCGA_data/blca_whole_slide_single_attn_visualization/train/smoothed_head_max'
save_dir = '/media/20t/BLCA/combine_attn_images'
# mid_dir = r'D:\P__Gene\midFiles\dino_wsi_attn_vis_allgene'

if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_list.sort()
#    image_list = image_list[180:200] + image_list[300:310]
    #random.shuffle(image_list)
    for image_path in tqdm(image_list[::-1]):
        # image = cv2.imread(image_path)
        # image_res = cv2.transpose(image)
        # cv2.imwrite(image_path, image_res)
        # continue
        img_name = os.path.basename(image_path)[:os.path.basename(image_path).find('.')]
        mask_path_list = glob.glob(os.path.join(mask_dir, img_name + '*.png'))
        #random.shuffle(mask_path_list)
        if len(glob.glob(os.path.join(save_dir, img_name + '*.jpg'))) == 186:
            continue
        if len(mask_path_list) > 0:
            print(image_path)
            try:
                img = cv2.imread(image_path)
            except Exception as e:
                print(e)
            # add_mask = cv2.imread(mask_path_list[0])
            # if not np.log2(round(img.shape[0]/add_mask.shape[0])).is_integer():
            #     raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
            #                 '{}'.format(round(img.shape[0]/add_mask.shape[0])))
            for mask_path in tqdm(mask_path_list):
                if os.path.exists(os.path.join(save_dir, os.path.basename(mask_path).replace('.png', '.jpg'))):
                    continue
                mask = cv2.imread(mask_path)
                # add_mask = add_mask + mask
                mid_size = (round(img.shape[1]), round(img.shape[0]))
                # resize_img = cv2.resize(img, mid_size, interpolation=cv2.INTER_NEAREST)
                # resize_mask = cv2.resize(mask, mid_size, interpolation=cv2.INTER_NEAREST)
                resize_mask = cv2.resize(mask, mid_size)
                assert resize_mask.shape == img.shape
                # try:
                # res_small = cv2.addWeighted(resize_img, 0.5, mask, 0.5, 0)
                # res_big = cv2.addWeighted(resize_mask, 0.5, img, 0.5, 0)
                res_mid = cv2.addWeighted(resize_mask, 0.4, img, 0.6, 0)
                # except:
                #     continue
                # cv2.imwrite(os.path.join(mid_dir, img_name +  '_mask.png'), add_mask)
                # cv2.imwrite(os.path.join(save_dir, img_name + '_big.png'), res_big)
                # cv2.imwrite(os.path.join(save_dir, img_name + '_small.png'), res_small)
                cv2.imwrite(os.path.join(save_dir, os.path.basename(mask_path).replace('.png', '.jpg')), res_mid, [int(cv2.IMWRITE_JPEG_QUALITY),70])
                # if os.path.getsize(mask_path) > 10240:
                #     shutil.copy(os.path.join(save_dir, os.path.basename(mask_path).replace('.png', '_mid.png')), os.path.join(save_dir, 'select'))

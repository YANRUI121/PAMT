#!/bin/env python
#coding:utf-8
#Author:itxx00@gmail.com
import os
import glob
import random

file_dir = '/home/sdg/TCGA_download/LUAD_SVS/patch_data_2000_20X_256/valid'

wsi_file_list = os.listdir(file_dir)
for wsi_file in wsi_file_list:
    wsi_img_list = glob.glob(os.path.join(os.path.join(file_dir, wsi_file), '*.png'))
    print(wsi_file)
    print(len(wsi_img_list))
    if len(wsi_img_list) > 2000:
        select_file_list = random.sample(wsi_img_list, len(wsi_img_list) - 2000)
        for select_file in select_file_list:
            os.remove(select_file)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openslide
import argparse
import glob
import cv2
import os
import random
import json
import torch
import math
from tqdm import tqdm


parser = argparse.ArgumentParser(description='K-menas for json file')
parser.add_argument('--wsi_attn_dir', default='/home/sda/zxy/luad_whole_slide_attn_single_txt/train_80_80_80/head', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--tissue_mask_dir', default='/home/sdg/TCGA_download/LUAD_SVS/tissue_data', metavar='TISSUE_MASK_PATH', type=str,
                    help='Path to the input tissue mask file')
parser.add_argument('--wsi_dir', default='/home/sdg/TCGA_download/LUAD_SVS/svs_data', metavar='WSI_PATH', type=str,
                    help='Path to the input wsi file')
parser.add_argument('--position_cluster_txt', default='/home/sdd/zxy/TCGA_data/luad_whole_slide_select_txt_dino', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--txt_path', default='/home/sda/zxy/luad_whole_slide_single_attn_visualization/train/gray_npy/head', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--png_path', default='/home/sda/zxy/luad_whole_slide_single_attn_visualization/train/gray_npy/head', metavar='PNG_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--class_num', default=50, metavar='CLASS_NUM', type=int, help='Clustering Number Class')
parser.add_argument('--slide_level', default=20, metavar='SLIDE_LEVEL', type=int, help='Coordinate reduction factor')
parser.add_argument('--patch_size', default=256, metavar='PATCH_SIZE', type=int, help='Coordinate reduction factor')

def decode_pos_txt(txt_path):
    feat_list = []
    with open(txt_path, 'r') as f:
        data_list = f.read().splitlines()
        for data in data_list:
            data = data.split('\t')
            for data_idx in data:
                feat_list.append(data_idx)
    f.close()
    return feat_list


def load_position_attn_data(cluster_position_file_list, wsi_attn_dir):
    pos_attn_dict = {}
    patch_count = 0
    for idx_pos, cluster_position_file in enumerate(cluster_position_file_list):
        patch_count += 500
        pos_data = decode_pos_txt(cluster_position_file)
        feat_data = torch.load(os.path.join(wsi_attn_dir, os.path.basename(cluster_position_file).replace('.txt', '.pth')))
        for idx, pos in enumerate(pos_data):
            if pos in pos_attn_dict:
                pos_attn_dict[pos].append(feat_data[:,idx])
            else:
                pos_attn_dict[pos] = [feat_data[:,idx]]
    return pos_attn_dict, patch_count


def process_pos_attn_dict(wsi_pos_attn_dict):
    for key, value in wsi_pos_attn_dict.items():
        if len(value) > 1:
            wsi_pos_attn_dict[key] = torch.mean(torch.stack(value, dim=0), dim=0)
        else:
            wsi_pos_attn_dict[key] = value[0]
    return wsi_pos_attn_dict

def draw_wsi_attn_sum(wsi_name, out_size, wsi_pos_attn_dict, save_attn_dir):
    res_mask = np.zeros(out_size)
    for key, values in wsi_pos_attn_dict.items():
        coords = key[1:-1].split(',')
        coord_x = round(int(coords[0])/512)
        coord_y = round(int(coords[1])/512)
        res_mask[coord_x, coord_y] = values.sum()
    res_mask = res_mask - np.min(res_mask)
    res_mask = res_mask / np.max(res_mask)
    res_map = cv2.applyColorMap(np.uint8(255*res_mask), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_attn_dir, wsi_name + '_sum.png'), res_map)

def draw_wsi_attn(wsi_name, out_size, scale_ratio, wsi_pos_attn_dict, save_attn_dir):
    for i in tqdm(range(186)):
        #import pdb;pdb.set_trace()
        res_mask = np.zeros(out_size)
        for key, values in wsi_pos_attn_dict.items():
            coords = key[1:-1].split(',')
            coord_x = round(int(coords[0])/scale_ratio)
            coord_y = round(int(coords[1])/scale_ratio)
            res_mask[coord_x, coord_y] = values[i]
        res_mask = res_mask - np.min(res_mask)
        res_mask = res_mask / np.max(res_mask)
        np.save(os.path.join(save_attn_dir, wsi_name + '_{}.npy'.format(str(i))), res_mask)
        res_map = cv2.applyColorMap(np.uint8(255*res_mask), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_attn_dir, wsi_name + '_{}.png'.format(str(i))), res_map)


def get_out_size(args, wsi_name):
    tissue_mask_path_list = glob.glob(os.path.join(args.tissue_mask_dir, wsi_name + '.' + '*.npy'))
    assert len(tissue_mask_path_list) == 1
    wsi_path_list = glob.glob(os.path.join(args.wsi_dir, wsi_name + '.' + '*.svs'))
    assert len(wsi_path_list) == 1
    slide = openslide.OpenSlide(wsi_path_list[0])
    mpp = round( float( slide.properties[openslide.PROPERTY_NAME_MPP_X] ), 1 ) 
    if mpp == 0.1:
        max_level = 80
    elif mpp == 0.3 or mpp == 0.2:
        max_level = 40
    elif mpp == 0.5:
        max_level = 20
    else:
        raise ValueError('当前分辨率为{}，不存在该倍数设置'.format(str(mpp)))
    X_slide, Y_slide = slide.level_dimensions[0]
    top_rate = int(max_level / args.slide_level)
    cur_level = round( top_rate / slide.level_downsamples[1] )
    patch_size = round(top_rate / slide.level_downsamples[cur_level]) * args.patch_size
    mask = np.load(tissue_mask_path_list[0])
    X_mask, Y_mask = mask.shape
    rate = round(X_slide/X_mask)
    out_size = (math.ceil(X_mask/(patch_size/rate)), math.ceil(Y_mask/(patch_size/rate)))
    print('当前图像最高分辨率为{}倍，组织掩码缩小{}倍，组织掩码尺寸为{}，输出图像尺寸为{}'.format(str(max_level), str(rate), str(mask.shape), str(out_size)))
    return out_size, round(patch_size*round(slide.level_downsamples[cur_level]))


def run(args):
    for idx in range(16):
        print('正在处理第{}个注意力头的特征文件'.format(str(idx)))
        wsi_attn_dir = args.wsi_attn_dir + str(idx)
        txt_path = args.txt_path + str(idx)
        print(wsi_attn_dir)
        print(txt_path)
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)
        all_attn_file_list = glob.glob(os.path.join(wsi_attn_dir, '*.pth'))
        print(all_attn_file_list[0])
        wsi_name_list =  list(set(map(lambda x: os.path.basename(x)[:os.path.basename(x).find('.')], all_attn_file_list)))
        wsi_name_list.sort()
        for idx_wsi, wsi_name in enumerate(wsi_name_list):
            cluster_position_file_list = glob.glob(os.path.join(args.position_cluster_txt, wsi_name + '.' + '*.txt'))
            wsi_attn_file_list = glob.glob(os.path.join(wsi_attn_dir, wsi_name + '.'  + '*.pth'))
            assert len(cluster_position_file_list) == len(wsi_attn_file_list) 
            print('正在处理第 {}/{} 个WSI文件:{}, 其含有{}个聚类文件'.format(str(idx_wsi+1),  len(wsi_name_list), wsi_name, str(len(cluster_position_file_list))))
            wsi_pos_attn_dict_repeat, patch_count = load_position_attn_data(cluster_position_file_list, wsi_attn_dir)
            wsi_pos_attn_dict = process_pos_attn_dict(wsi_pos_attn_dict_repeat)
            print('当前文件去重之前有{}个patch，去重之后有{}个patch'.format(str(patch_count), str(len(wsi_pos_attn_dict.keys()))))
            #torch.save(wsi_pos_attn_dict, os.path.join(txt_path, wsi_name + '.pth'))
            out_size, scale_ratio = get_out_size(args, wsi_name)
            draw_wsi_attn(wsi_name, out_size, scale_ratio, wsi_pos_attn_dict, txt_path)
            #draw_wsi_attn_sum(wsi_name, out_size, wsi_pos_attn_dict, args.txt_path)


def main():
    args = parser.parse_args()
    run(args)      


if __name__ == '__main__':
    main()

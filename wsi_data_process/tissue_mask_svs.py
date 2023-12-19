import sys
import os
import argparse
import logging
import glob

import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('--wsi_path', default='/home/sdg/TCGA_download/LUAD_SVS/svs_data', metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('--npy_path', default='/home/sdg/TCGA_download/LUAD_SVS/tissue_data', metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--gray_path', default='/home/sdg/TCGA_download/LUAD_SVS/tissue_data', metavar='GRAY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=2, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')

#转换颜色空间，ostu阈值分割，开闭形态学运算
def run(args):
    logging.basicConfig(level=logging.INFO)

    paths = glob.glob(os.path.join(args.wsi_path, '*.svs'))
    for path in paths:
        print(path)
        npy_name = os.path.basename(path)
        npy_path = os.path.join(args.npy_path,npy_name.replace('svs','npy'))
        if os.path.exists(npy_path):
            continue
        #import pdb;pdb.set_trace()
        slide = openslide.OpenSlide(path)
        print(slide.level_count)
        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                               min(args.level,slide.level_count-1),
                               slide.level_dimensions[min(args.level,slide.level_count-1)]).convert('RGB')),
                               axes=[1, 0, 2])
        slide.close()
        img_HSV = rgb2hsv(img_RGB)

        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > args.RGB_min
        min_G = img_RGB[:, :, 1] > args.RGB_min
        min_B = img_RGB[:, :, 2] > args.RGB_min

        #tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
        tissue_mask = tissue_RGB & min_R & min_G & min_B
        np.save(npy_path, tissue_mask)
        plt.imsave(os.path.join(args.gray_path, npy_name.replace('svs','png')), tissue_mask)

def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

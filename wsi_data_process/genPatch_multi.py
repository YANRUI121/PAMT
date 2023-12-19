import os
import argparse
import time
import logging
import numpy as np
import openslide
import random
from multiprocessing import Pool, Value, Lock

parser = argparse.ArgumentParser(description='Generate the patch of tumor')
parser.add_argument('--wsi_path', default='/home/sdg/TCGA_download/LUAD_SVS/svs_data', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--mask_path', default='/home/sdg/TCGA_download/LUAD_SVS/tissue_data', metavar='MASK_PATH', type=str,
                    help='Path to the tumor mask of the input WSI file')
parser.add_argument('--patch_path', default='/home/sdg/TCGA_download/LUAD_SVS/patch_data_whole_slide_20X_256', metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=20, type=int, help='patch level, '
                    'default 20')
parser.add_argument('--num_process', default=10, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()

def process(opts):
    i, wsi_path, patch_dir, x, y, rate, args, patch_size, cur_level = opts
    #x = int(int(x_center) - args.patch_size * rate / 2)
    #y = int(int(y_center) - args.patch_size * rate / 2)
    #img_path = os.path.join(patch_dir, os.path.basename(wsi_path)[:-4] + '_' + str(x) + '_' + str(y) + '.png')
    #if os.path.exists(img_path):
    #    continue
    slide = openslide.OpenSlide(wsi_path)
    #print(x_center, y_center)
    #print(x, y)
    #import pdb;pdb.set_trace()
    img = slide.read_region(
        (x, y), cur_level,
        (patch_size, patch_size)).convert('RGB')

    if patch_size != args.patch_size:
        img = img.resize((args.patch_size, args.patch_size))
    img_path = os.path.join(patch_dir, os.path.basename(wsi_path)[:-4] + '_' + str(x) + '_' + str(y) + '.png')
    img.save(img_path)
    #print(os.path.getsize(img_path))
    #if os.path.getsize(img_path) < 102400:
    #    os.remove(img_path)

    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args):
    logging.basicConfig(level=logging.INFO)
    ff = os.walk(args.wsi_path)
    paths = []
    opts_list = []
    for root, dirs, files in ff:
        for file in files:
            if os.path.splitext(file)[1] == '.svs':
                paths.append(os.path.join(root, file))
    num_svs = 0
    for idx_path, path in enumerate(paths):
        wsi_name = os.path.basename(path)[:-4]
        #if not wsi_name=='TCGA-XF-A9SW-01Z-00-DX1.3FC62DD6-E5D2-4F74-A937-A2A62946E5C0':
        #name_position = len('TCGA-FD-A3N6-01Z-00-DX1')
        ##temp_list = ['TCGA-XF-A9SW-01Z-00-DX1', 'TCGA-FD-A3N6-01Z-00-DX1', 'TCGA-XF-A8HB-01Z-00-DX1', 'TCGA-FD-A43N-01Z-00-DX1' ,'TCGA-4Z-AA86-01Z-00-DX1', 'TCGA-FJ-A3Z7-01Z-00-DX4']
        #temp_list = ['TCGA-2F-A9KQ-01Z-00-DX1', 'TCGA-E7-A97Q-01Z-00-DX1', 'TCGA-S5-AA26-01Z-00-DX1']
        #if wsi_name[:name_position] not in temp_list:
        #    continue
        patch_dir = os.path.join(args.patch_path, wsi_name)
        if  os.path.exists(patch_dir):
            #continue
            pass
        else:
            os.mkdir(patch_dir)
        mask_path = os.path.join(args.mask_path, wsi_name + '.npy')
        if not os.path.exists(mask_path):
            continue
        print('Processing {}/{} file : {}'.format(idx_path+1, len(paths), wsi_name))
        mask = np.load(mask_path)
        slide = openslide.OpenSlide(path)
        try:
            mpp = round( float( slide.properties[openslide.PROPERTY_NAME_MPP_X] ), 1 )
        except:
            mpp = 0.3
        if mpp == 0.1:
            max_level = 80
        elif mpp == 0.3 or mpp == 0.2:
            max_level = 40
        elif mpp == 0.5:
            max_level = 20
        else:
            raise ValueError('当前分辨率为{}，不存在该倍数设置'.format(str(mpp)))
        # level = slide.level_count
        X_slide, Y_slide = slide.level_dimensions[0]
        #x_level, y_level = slide.level_dimensions[args.level]
        rate = round(max_level // args.level)
        cur_level = round(rate / slide.level_downsamples[1] )
        patch_level = round(X_slide / mask.shape[0])
        patch_size = round(rate / slide.level_downsamples[cur_level]) * args.patch_size
        step = int(args.patch_size / patch_level)
        #print(path, slide.level_count, rate, X_slide, Y_slide)
        print('当前图像最高倍数为{}倍，level1图像倍数为{}倍，取图倍数为{}倍，图像尺寸为{}'.format(max_level, slide.level_downsamples[1], cur_level, patch_size))
        slide.close()
        X_mask, Y_mask = mask.shape

        #if X_slide // X_mask != Y_slide // Y_mask:
        #    raise Exception('Slide/Mask dimension does not match ,'
        #                    'X_slide / X_mask: {} / {}, '
        #                    'Y_slide / Y_mask: {} / {}'
        #                    .format(X_slide, X_mask, Y_slide, Y_mask))


        #resolution = X_slide // X_mask
        # print(path, level, X_slide, Y_slide,X_mask, Y_mask, resolution)
        #resolution_log = np.log2(resolution)
        #if not resolution_log.is_integer():
        #    raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
        #                    '{}'.format(resolution))

        # all the idces for tissue region from the tissue mask
        #X_idcs, Y_idcs = np.where(mask)
        ori_x, ori_y = list((map(lambda x: np.int32(x/rate/step), np.where(mask))))
        coords = list(set(zip(ori_x, ori_y)))
        #for idc_num in range(len(X_idcs)):
        #    temp = [X_idcs[idc_num], Y_idcs[idc_num]]
        #    coords.append(temp)
        random.shuffle(coords)
        #print('{}/{}: {}-count of coords:{}, level 0 is {} times tissue, divide {} times, cut patch size:{}'.format(
        #    str(num_svs + 1), str(len(paths)), wsi_name[:wsi_name.find('DX')], len(coords), patch_level, str(rate*step), patch_size))
        num_svs += 1
        #import pdb;pdb.set_trace()
        num = 0
        #for idx in range(min(2000,len(coords))):
        #slide = openslide.OpenSlide(path)
        for idx in range(len(coords)):
        #for idx in range(min(2000,len(coords))):
            x_mask, y_mask = coords[idx][0], coords[idx][1]
            x_center = int((x_mask + 0.5) * patch_level * step * rate)
            y_center = int((y_mask + 0.5) * patch_level * step * rate)
            x = int(int(x_center) - args.patch_size * rate / 2)
            y = int(int(y_center) - args.patch_size * rate / 2)
            img_path = os.path.join(patch_dir, wsi_name + '_' + str(x) + '_' + str(y) + '.png')
            if os.path.exists(img_path):
                continue
                #if (x_center < 0) | (y_center<0):
                #    import pdb;pdb.set_trace()
                #    print(x_center, y_center)
    #print(x_center, y_center)
    #print(x, y)
    #import pdb;pdb.set_trace()
            #try:
            #    img = slide.read_region(
            #            (x, y), cur_level,
            #            (patch_size, patch_size)).convert('RGB')
            #except:
            #    continue

            #if patch_size != args.patch_size:
            #    img = img.resize((args.patch_size, args.patch_size))
            #img_path = os.path.join(patch_dir, wsi_name + '_' + str(x) + '_' + str(y) + '.png')
            #img.save(img_path)
    #print(os.path.getsize(img_path))
    #if os.path.getsize(img_path) < 102400:
            opts_list.append((num, path, patch_dir, x, y, rate, args, patch_size, cur_level))
            #print(path)
            #print(x, y)
            num = num + 1
    #print(opts_list)
    print(len(opts_list))
    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)
    process(opts_list)



def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

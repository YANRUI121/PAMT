import openslide
import glob
import os
import argparse
import imageio
import numpy as np

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('--wsi_path', default='/home/sdg/TCGA_download/LUAD_SVS/svs_data', metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('--npy_path', default='/home/sdc/tcga_sdd/luad_level1', metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=1, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
def run(args):
    ff = os.walk(args.wsi_path)
    paths = []
    for root, dirs, files in ff:
        for file in files:
            if os.path.splitext(file)[1] == '.svs':
                paths.append(os.path.join(root, file))
    for path in paths:
	    print(path)
	    slide = openslide.OpenSlide(path)
	    print(slide.level_count)
	    # note the shape of img_RGB is the transpose of slide.level_dimensions
	    img_RGB = np.transpose(np.array(slide.read_region((0, 0),
	    #img_RGB = np.array(slide.read_region((0, 0),
	                           min(args.level, slide.level_count-1),
	                           slide.level_dimensions[min(args.level, slide.level_count-1)]).convert('RGB')), 
	                           axes=[1, 0, 2])
	    slide.close()
	    npy_name = os.path.basename(path)
	    imageio.imsave(os.path.join(args.npy_path, npy_name[:-4] + '.jpg'),img_RGB)

def main():
	args = parser.parse_args()
	run(args)


if __name__ == '__main__':
    main()

import os
import argparse
import glob
import json
import sklearn.utils as su
from tqdm import tqdm
from multiprocessing import Process,Manager,Value,Pool


parser = argparse.ArgumentParser(description='record selected features for every wsi')
parser.add_argument('--feature_txt_path', type=str, default='/home/sdd/zxy/TCGA_data/luad_all_dino_feat_txt',
                    help='path to feature txt')
parser.add_argument('--result_txt_path', type=str, default='/home/sdd/zxy/TCGA_data/luad_whole_slide_select_txt_dino',
                    help='path to result txt')
parser.add_argument('--write_txt_dir', type=str, default='/home/sdd/zxy/TCGA_data/luad_whole_slide_select_feat_txt_dino',
                    help='path to result txt')
parser.add_argument('--process_count', type=int, default=10, help='the count of multiprocess')


def decode_feature_txt(feat_txt):
    file_object = open(feat_txt,'r')
    feat_dict = {}
    for line in file_object.readlines():
        feat_dict.update(json.loads(line))
    return feat_dict


def load_data(txt_path, feat_data):
    feat_list = []
    with open(txt_path, 'r') as f:
        #data_list = f.readlines()
        data_list = f.read().splitlines()
        for data in data_list:
            data = data.split('\t') 
            for data_idx in data:
                if feat_data.get(data_idx):
                    feat_list.append(feat_data[data_idx])
    f.close()
    assert len(feat_list)==500
    #print(len(feat_list))
    return feat_list


def get_file_name(args, feat_dir):
    cls_list = glob.glob(os.path.join(feat_dir, '*_0.txt'))
    #gene_type = os.path.split(feat_dir)[-1]
    for i, cls_path in enumerate(cls_list):
        cls_name = os.path.basename( cls_path )
        if os.path.exists(os.path.join(args.write_txt_dir, cls_name)):
            continue
        txt_name = cls_name[:cls_name.find( 'kmeans' )]
        feat_path = os.path.join( args.feature_txt_path, txt_name + '.txt' )
        feat_data = decode_feature_txt(feat_path)
        wsi_select_count = len(glob.glob(os.path.join(feat_dir, txt_name + '*.txt')))
        print('Precessing {}/{} file:{}, all feat count:{}.'.format(str(i+1),  len(cls_list), txt_name, str(wsi_select_count)))
        for j in tqdm(range(wsi_select_count)):
            cls_path = cls_path[:cls_path.rfind('_')] + '_{}.txt'.format(str(j))
            cls_name = os.path.basename( cls_path )
            write_path = os.path.join(args.write_txt_dir, cls_name)
            if os.path.exists(write_path):
                continue
            all_feat_list = load_data(cls_path, feat_data)
            with open(write_path, 'w') as f:
                json.dump( all_feat_list, f)
    return list(all_feat_list)


def main():
    args = parser.parse_args()
    get_file_name(args, args.result_txt_path)

if __name__ == '__main__':
    main()

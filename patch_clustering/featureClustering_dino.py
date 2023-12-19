from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from skfuzzy.cluster import cmeans
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import glob
import os
import random
import json
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser(description='K-menas for json file')
parser.add_argument('--feat_path', default='/home/sdd/zxy/TCGA_data/luad_all_dino_feat_txt', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--position_path', default='/home/sdd/zxy/TCGA_data/luad_all_dino_feat_txt', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
#parser.add_argument('--new_feat_path', default='/home/sdd/zxy/TCGA_data/featTxt/tcga_all_2_cancer_0402_pca_50/wild', metavar='WSI_PATH', type=str,
#                    help='Path to the input WSI file')
parser.add_argument('--save_txt_path', default='/home/sdd/zxy/TCGA_data/luad_all_dino_feat_txt', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--txt_path', default='/home/sdd/zxy/TCGA_data/luad_all_dino_feat_clustering', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--png_path', default='/home/sdd/zxy/TCGA_data/luad_all_dino_feat_clustering', metavar='PNG_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--class_num', default=50, metavar='CLASS_NUM', type=int, help='Clustering Number Class')
parser.add_argument('--scale_ratio', default=512, metavar='SCALE_RATIO', type=int, help='Coordinate reduction factor')

random = np.random.RandomState(0)

def randomcolor(class_num):
    colorList = []
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    for num in range(class_num):
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,14)]
        colorList.append("#"+color)
    return colorList


def loadDataSet(fileName):
    with open(fileName, 'r')as fp:
        json_data = fp.readlines()
        featName = []
        feats = []
        for line in json_data:
            lineList = line.split(':')
            featName.append(lineList[0][2:-1])
            #new_features = pac_train(list(map(float, lineList[1][2:-3].split(', '))), cls_nums)
            feats.append(list(map(float, lineList[1][2:-3].split(', '))))
    return featName, feats


def load_position(position_path):
    wsi_count = []
    with open(position_path, "r") as f:
        content = f.read()
        path_data = json.loads(content)
    wsi_name_list= list(map(lambda x: os.path.split(os.path.dirname(x))[-1], path_data))
    patch_name_list = list(map(lambda x: os.path.basename(x)[:-4], path_data))
    patch_position_list = list(map(lambda x: x[x.find('_') + 1:], patch_name_list))
    temp_wsi_name = wsi_name_list[0]
    for patch_idx, wsi_name in enumerate(wsi_name_list):
        if temp_wsi_name != wsi_name_list[patch_idx]:
            wsi_count.append(patch_idx)
            temp_wsi_name = wsi_name_list[patch_idx]
    wsi_count.append(len(wsi_name_list))
    return wsi_count, wsi_name_list, patch_position_list
    

def showClass(args, json_name, feat_name, features, labels, color_list, ext=''):
    png_file = os.path.join(args.png_path, json_name.replace('.txt', ext + '.png'))
    #color_list = randomcolor(args.class_num)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i, coords in enumerate(feat_name):
        coord = coords.split(',')
        plt.plot(int(coord[0][1:]), int(coord[1][1:-1]), color=color_list[labels[i]], marker='.', markersize=4)

    plt.savefig(png_file, bbox_inches='tight', dpi=1000)
    plt.clf()

def showEvaluate(args, feat_name, features, labels, json_name, ext):
    #if len(features) > args.class_num:
        #print('silhoutte: ',metrics.silhouette_score(features, labels))
        #print('calinski: ',metrics.calinski_harabasz_score(features, labels))
        #print('davies: ',metrics.davies_bouldin_score(features, labels))
    data = (list(zip(feat_name, labels)))
    data = pd.DataFrame(data)
    json_name = json_name.replace('.txt', ext + '_cls.txt')
    txt_path = os.path.join(args.txt_path, json_name)
    data.to_csv(txt_path, sep='\t', index=0, header=0)

# K-Means聚类
def kmeans_train(features, clusters):
    return KMeans(n_clusters=clusters).fit(features)

# 谱聚类
def spectral_train(features, clusters):
    return SpectralClustering(n_clusters=clusters).fit(features)

#模糊C均值聚类
def fcm_train(features, clusters):
    feature_t = np.array(features).T
    center, u, u0, d, jm, p, fpc = cmeans(feature_t, m=2, c=clusters, error=0.0001, maxiter=1000)
    return np.argmax(u, axis=0)  # 取得列的最大值

def pca_train(features, cls_num):
    pca = PCA(n_components=cls_num,svd_solver="arpack")
    new_feature = pca.fit_transform(features)
    return new_feature

def run1(args):
    paths = glob.glob(os.path.join(args.feat_path, '*.txt'))
    color_list = randomcolor(args.class_num)
    paths.sort()
    #paths.reverse()

    for path in paths:
        print(path)
        json_name = os.path.basename(path)
        #if os.path.exists(os.path.join(args.png_path, json_name.replace('.txt', 'kmeans_cls.txt'))):
        if os.path.exists(os.path.join(args.png_path, json_name.replace('.txt', 'kmeans.png'))):
            continue
        feat_name, features = loadDataSet(path)
        if len(features)<1:
            continue
        cls_nums = args.class_num if len(features)>=args.class_num else len(features)
        #cls_nums = 50
        #print(cls_nums)
        #new_features = pca_train(features, cls_nums)
        #featureDict = {}
        
        #new_feat_path = os.path.join(args.new_feat_path, json_name)
        #with open(new_feat_path, 'w') as f:
        #    for i in range(len(feat_name)):
        #        featureDict['{}'.format(str(feat_name[i]))] = new_features[i].tolist()
        #    json.dump(featureDict, f)
        #    f.write('\n')

        print('Clustering....')
        #kmeans_cls = kmeans_train(features, cls_nums)
        kmeans_cls = kmeans_train(features, cls_nums)
        print('Writing...')
        #showEvaluate(args, feat_name, features, kmeans_cls.labels_, json_name, 'kmeans')
        showEvaluate(args, feat_name, features, kmeans_cls.labels_, json_name, 'kmeans')
        print('Ploting...')
        showClass(args, json_name, feat_name, features, kmeans_cls.labels_,color_list, 'kmeans')
        #print('Clustering....')
        #spectral_cls = spectral_train(features, cls_nums)
        #print('Writing...')
        #showEvaluate(args, feat_name, features, spectral_cls.labels_, json_name, 'spec')
        #print('Ploting...')
        #showClass(args, json_name, feat_name, features, spectral_cls.labels_, 'spec')

        #fcm_labels = fcm_train(features, cls_nums)
        #showEvaluate(args, features, fcm_labels, json_name, 'fcm')
        #showClass(args, json_name, features, fcm_labels, 'fcm')


def run2(args):
    wsi_count_list, wsi_name_list, patch_position_list = load_position(args.position_path)
    x_position_list = list(map(lambda x: int(x[:x.find('_')]), patch_position_list))
    y_position_list = list(map(lambda x: int(x[x.find('_') + 1:]), patch_position_list))
    patch_features = torch.load(args.feat_path)
    start_count = 0
    for wsi_idx, wsi_count in enumerate(wsi_count_list):
        print('{}/{} : {}, {}'.format(str(wsi_idx + 1), str(len(wsi_count_list)), wsi_name_list[start_count], str(wsi_count)))
        wsi_txt_path = os.path.join(args.save_txt_path, wsi_name_list[start_count] + '.txt')
        #if os.path.exists(wsi_txt_path):
        #    start_count = wsi_count
        #    continue
        with open(wsi_txt_path, "w") as f:
            idList = list(zip(x_position_list[start_count:wsi_count], y_position_list[start_count:wsi_count]))
            featureList = list(patch_features[start_count:wsi_count].tolist())
            for idNum,idName in tqdm(enumerate(idList)):
                featureDict = {}
                featureDict['{}'.format(str(idName))]=featureList[idNum]
                json.dump(featureDict, f)
                f.write('\n')
        start_count = wsi_count


def main():
    args = parser.parse_args()
    run1(args)      #特征聚类
    #run2(args)      #降torch的特征形式写为WSI的txt格式


if __name__ == '__main__':
    main()

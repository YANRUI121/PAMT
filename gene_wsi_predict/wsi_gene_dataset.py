from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import glob
import random
import json
import numpy as np

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, wsi_feature_dir, gene_feature_dir, cox_txt_path, max_dim = 10, mode='train', transform=None):
        self.wsi_feature_dir = wsi_feature_dir
        self.gene_feature_dir = gene_feature_dir
        self.cox_txt_path = cox_txt_path
        self.max_dim = max_dim
        self.mode = mode
        self.transform = transform
        #import pdb;pdb.set_trace()
        self.gene_file_list = glob.glob(os.path.join(self.gene_feature_dir, '*.txt'))
        self.gene_patient_list = list(map(lambda x:os.path.basename(x)[:12], self.gene_file_list))
        self.wsi_file_list = glob.glob(os.path.join(self.wsi_feature_dir, '*.txt'))
        self.wsi_patient_list = list(map(lambda x:os.path.basename(x)[:12], self.wsi_file_list))
        self.wsi_file_set = self.wsi_file_list
        self._pre_process()


    def _pre_process(self):
        with open(self.cox_txt_path ) as f:
            cox_time_list = f.read().splitlines()
        self.cox_dict = {}
        tcga_name_list = []
        for cox_time in cox_time_list:
            tcga_name, futime, fustat = cox_time.split( '\t' )
            tcga_name_list.append(tcga_name)
            self.cox_dict[tcga_name] = [futime, fustat]

        self.patient_list = list(set(tcga_name_list).intersection(set(self.gene_patient_list), set(self.wsi_patient_list)))
        if self.mode == 'train':
            random.shuffle(self.patient_list)


    def __len__(self):
        return len(self.patient_list)


    def __getitem__(self, index):
        patient_name = self.patient_list[index]
        #import pdb;pdb.set_trace()
        for gene_file in self.gene_file_list:
            if os.path.basename(gene_file)[:12] == patient_name:
                with open(gene_file, 'r') as f_gene:
                    gene_feat = np.array(json.load(f_gene))
                    break
        wsi_feat = []
        random.shuffle(self.wsi_file_set)
        for wsi_file in self.wsi_file_set:
            if os.path.basename(wsi_file)[:15] == os.path.basename(gene_file)[:15]:
                with open(wsi_file, 'r') as f_wsi:
                    wsi_feat = np.array(json.load(f_wsi))
                    break
                    #wsi_feat.append(np.array(json.load(f_wsi)))

                    #if len(wsi_feat) > 9:
                    #    break
        # wsi_feat = np.stack(wsi_feat)
        # wsi_feat = np.array(np.concatenate(wsi_feat, 0))
        if self.transform is not None:
            wsi_feat = self.transform(wsi_feat)
        wsi_feat = torch.Tensor(wsi_feat)
        gene_feat = torch.Tensor(gene_feat)
        #print(round(float(self.samples[index][1][0])), round(float(self.samples[index][1][1])))
        return wsi_feat, gene_feat, round(float(self.cox_dict[patient_name][0])), round(float(self.cox_dict[patient_name][1]))

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        wsi_feat, gene_feat, futime, fustat = tuple(zip(*batch))

        wsi_feat = torch.stack(wsi_feat, dim=0)
        gene_feat = torch.stack(gene_feat, dim=0)
        #import pdb;pdb.set_trace()
        futime = torch.Tensor(futime)
        fustat = torch.Tensor(fustat)
        return wsi_feat, gene_feat, futime, fustat

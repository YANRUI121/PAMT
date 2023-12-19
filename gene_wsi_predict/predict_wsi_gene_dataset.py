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
        self.select_wsi_file_list = []
        #import pdb;pdb.set_trace()
        self.gene_file_list = glob.glob(os.path.join(self.gene_feature_dir, '*.txt'))
        self.gene_patient_list = list(map(lambda x:os.path.basename(x)[:12], self.gene_file_list))
        self.wsi_file_list = glob.glob(os.path.join(self.wsi_feature_dir, '*.txt'))
        self.wsi_patient_list = list(map(lambda x:os.path.basename(x)[:12], self.wsi_file_list))
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

        self.patient_list = set(list(set(tcga_name_list).intersection(set(self.gene_patient_list), set(self.wsi_patient_list))))
        for wsi_file_path in self.wsi_file_list:
            if os.path.basename(wsi_file_path)[:12] in self.patient_list:
                self.select_wsi_file_list.append(wsi_file_path)


    def __len__(self):
        return len(self.select_wsi_file_list)


    def __getitem__(self, index):
        #import pdb;pdb.set_trace()
        wsi_file_path = self.select_wsi_file_list[index]
        patient_name = os.path.basename(wsi_file_path)[:12] 
        for gene_file in self.gene_file_list:
            if os.path.basename(gene_file)[:15] == os.path.basename(wsi_file_path)[:15]:
                with open(gene_file, 'r') as f_gene:
                    gene_feat = np.array(json.load(f_gene))
                    break
        with open(wsi_file_path, 'r') as f_wsi:
            wsi_feat = np.array(json.load(f_wsi))
        if self.transform is not None:
            wsi_feat = self.transform(wsi_feat)
        wsi_feat = torch.Tensor(wsi_feat)
        gene_feat = torch.Tensor(gene_feat)
        #print(round(float(self.samples[index][1][0])), round(float(self.samples[index][1][1])))
        return wsi_feat, gene_feat, round(float(self.cox_dict[patient_name][0])), round(float(self.cox_dict[patient_name][1])), os.path.basename(wsi_file_path)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        wsi_feat, gene_feat, futime, fustat, file_name = tuple(zip(*batch))

        wsi_feat = torch.stack(wsi_feat, dim=0)
        gene_feat = torch.stack(gene_feat, dim=0)
        #import pdb;pdb.set_trace()
        futime = torch.Tensor(futime)
        fustat = torch.Tensor(fustat)
        return wsi_feat, gene_feat, futime, fustat, file_name

#!/bin/env python
#coding:utf-8
#Author:itxx00@gmail.com
import os


cox_time_path = 'cox_time.txt'
train_pathway_dir = 'pathway_gene_features/train'
valid_pathway_dir = 'pathway_gene_features/valid'
svs_dir = '/home/sdc/pangguang/svs/old_svs'


def run():
    svs_file_list = os.listdir(svs_dir)
    train_pathway_file_list = os.listdir(train_pathway_dir)
    valid_pathway_file_list = os.listdir(valid_pathway_dir)
    #pathway_file_list = train_pathway_file_list + valid_pathway_file_list
    pathway_file_list = valid_pathway_file_list 
    with open(cox_time_path) as f:
        cox_time_list = f.read().splitlines()
    cox_file_list = []
    for cox_time in cox_time_list:
        tcga_name, futime, fustat = cox_time.split( '\t' )
        cox_file_list.append(tcga_name)

    svs_patient_list = list(map(lambda x: x[:12], svs_file_list))
    svs_patient_set = set(svs_patient_list)
    pathway_patient_list = list(map(lambda x: x[:12], pathway_file_list))
    pathway_patient_set = set(pathway_patient_list)
    cox_patient_list = list(map(lambda x: x[:12], cox_file_list))
    cox_patient_set = set(cox_patient_list)

    svs_pathway_cox_set = svs_patient_set & pathway_patient_set & cox_patient_set
    svs_count = 0
    pathway_count = 0
    cox_count = 0
    for file_name in svs_pathway_cox_set:
        if file_name in svs_patient_list:
            svs_count += svs_patient_list.count(file_name)
            pathway_count += pathway_patient_list.count(file_name)
            if pathway_patient_list.count(file_name) > 1:
                print(file_name)
            cox_count += cox_patient_list.count(file_name)
    print('三种文件都有的有{}个病人，其中有{}个svs文件，{}个基因文件，{}个生存期文件.'.format(len(svs_pathway_cox_set), svs_count, pathway_count, cox_count))


run()

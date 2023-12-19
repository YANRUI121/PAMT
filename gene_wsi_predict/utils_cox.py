import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import torch.nn.functional as F


def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]:
                        concord = concord + 1
                    elif hazards[j] < hazards[i]:
                        concord = concord + 0.5
    return (concord / total)


def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy().reshape(-1)
    hazards = hazards.cpu().numpy().reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)

    return (concordance_index(new_surv, -new_hazard, new_label))


def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    labels = labels.data.cpu().numpy()
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)



def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)
 
class Regularization(object):
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class NegativeLogLikelihood(nn.Module):
    def __init__(self, l2_reg):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = l2_reg
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, survtime, censor, model):
        mask = torch.ones(survtime.shape[0], survtime.shape[0])
        mask[(survtime.T - survtime) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0)/torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * censor)/torch.sum(censor)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss


def CoxLoss(survtime, censor, hazard_pred, loss='cox-nnet', model=None, l2_reg=1e-2):
    if loss == 'deepsurv':
        nll_loss = NegativeLogLikelihood(l2_reg)
        return nll_loss(hazard_pred, survtime, censor, model)
    elif loss == 'cox-nnet':
        #import pdb;pdb.set_trace()
        current_batch_len = len(survtime)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
           for j in range(current_batch_len):
              if (survtime[j] >= survtime[i]): R_mat[i, j] = 1

        R_mat = torch.FloatTensor(R_mat)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        R_mat = R_mat.cuda()
        exp_theta = exp_theta.cuda()
        censor = censor.cuda()
        theta = theta.cuda()
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss_cox


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def train_one_epoch(model, topK, criterion, optimizer, data_loader, epoch, reg_loss=False, train_flag=0, contrastive_loss_flag=0):
    #import pdb;pdb.set_trace()
    model.train()
    #regularization_l1_loss = 0
    #if l1_loss:
    #    for param in model.parameters():
    #        if param.requires_grad:
    #            regularization_l1_loss += torch.sum(torch.abs(param)).cuda()
    accu_loss = torch.zeros(1).cuda()  # 累计损失
    #optimizer.zero_grad()

    sample_num = 0

    pred_all = None
    survtime_all = []
    status_all = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    #import pdb;pdb.set_trace()
    for step, data in enumerate(data_loader):

        wsi_features, gene_features, futime, fustat = data
        sample_num += gene_features.shape[0]
        labels = torch.zeros(gene_features.shape[0], gene_features.shape[1], wsi_features.shape[1])
        labels[:, :, :topK] = 1
        labels = labels.cuda()

        if train_flag==0:
            if contrastive_loss_flag == 1:
                gene2wsi_feature, pred = model(wsi_features, gene_features)
                #gene2wsiloss, pred = model(wsi_features, gene_features)
                sorted_gen2wsi_feat, _ = torch.sort(gene2wsi_feature, descending=True, dim=2)
                gene2wsiloss = criterion(sorted_gen2wsi_feat, labels)
            else:
                pred, _ = model(wsi_features, gene_features)
                gene2wsiloss=0
        elif train_flag==1:
            pred = model(wsi_features)
        survtime_all.append(np.squeeze(futime.data.cpu().numpy()))  # if time are days
        status_all.append(np.squeeze(fustat.data.cpu().numpy()))

        if step == 0:
            pred_all = pred
            survtime_torch = futime
            fustat_torch = fustat
        else:
            fustat_torch = torch.cat( [fustat_torch, fustat] )
            pred_all = torch.cat( [pred_all, pred] )
            survtime_torch = torch.cat([survtime_torch, futime])

        if (reg_loss) & (train_flag==0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss + reg_loss(model) 
        elif (reg_loss) & (train_flag==1):
            loss = CoxLoss(futime, fustat, pred) + reg_loss(model) 
        elif (~reg_loss) & (train_flag==0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss 
        elif (~reg_loss) & (train_flag==1):
            loss = CoxLoss(futime, fustat, pred)
        #else:
        #    loss = CoxLoss(futime, fustat, pred) + gene2wsiloss 

        loss=loss.mean()
        #optimizer.step()
        optimizer.zero_grad()

        #loss.backward(torch.ones_like(loss))
        loss.sum().backward()
        #loss.backward()
        optimizer.step()
        #loss.backward(retain_graph=True)
        accu_loss += loss.detach()



        data_loader.desc = "[train epoch {}] loss: {:.6f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)


    acc = accuracy_cox(pred_all.data, fustat_torch)
    pvalue_pred = cox_log_rank(pred_all.data, fustat_torch, survtime_torch)
    c_index = CIndex_lifeline(pred_all.data, fustat_torch, survtime_torch)
    return accu_loss.item() / (step + 1), acc, pvalue_pred, c_index


@torch.no_grad()
def evaluate(model, topK, criterion, data_loader, epoch, json_path, reg_loss=False, train_flag=0, contrastive_loss_flag=0):
    model.eval()
    #regularization_l1_loss = 0
    #if l1_loss:
    #    for param in model.parameters():
    #        if param.requires_grad:
    #            regularization_l1_loss += torch.sum(torch.abs(param)).cuda()

    accu_loss = torch.zeros(1).cuda()  # 累计损失

    sample_num = 0
    pred_all = None
    survtime_all = []
    status_all = []

    # f = open(json_path, 'w')
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        wsi_features, gene_features, futime, fustat = data
        sample_num += gene_features.shape[0]
        labels = torch.zeros(gene_features.shape[0], gene_features.shape[1], wsi_features.shape[1])
        labels[:, :, :topK] = 1
        labels = labels.cuda()

        if train_flag == 0:
            if contrastive_loss_flag==1:
                #gene2wsiloss, pred = model(wsi_features, gene_features,)
                gene2wsi_feature, pred = model(wsi_features, gene_features,)
                sorted_gen2wsi_feat, _ = torch.sort(gene2wsi_feature, descending=True, dim=2)
                gene2wsiloss = criterion(sorted_gen2wsi_feat, labels)
            else:
                pred, _ = model(wsi_features, gene_features,)
                gene2wsiloss=0
        elif train_flag == 1:
            pred = model(wsi_features)
        # for index, wsi_path_sample in enumerate(wsi_path):
        #     f.write(wsi_path_sample + '\n')
        #     f.write(str(pred_features[index].cpu().tolist()) + '\n')
            #f.write(str(pred[index].item())+ '\n')
        survtime_all.append( np.squeeze( futime.data.cpu().numpy() ) )  # if time are days
        status_all.append( np.squeeze( fustat.data.cpu().numpy() ) )
        if step == 0:
            pred_all = pred
            survtime_torch = futime
            fustat_torch = fustat
        else:
            fustat_torch = torch.cat( [fustat_torch, fustat] )
            pred_all = torch.cat( [pred_all, pred] )
            survtime_torch = torch.cat( [survtime_torch, futime] )
        #loss = CoxLoss(futime, fustat, pred) + gene2wsiloss + 0.01 * regularization_l1_loss
        if reg_loss & (train_flag==0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss + reg_loss(model) 
        elif reg_loss & (train_flag==1):
            loss = CoxLoss(futime, fustat, pred) + reg_loss(model) 
        elif (~reg_loss) & (train_flag==0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss 
        elif (~reg_loss) & (train_flag==1):
            loss = CoxLoss(futime, fustat, pred) 
        loss= loss.mean()
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.6f}".format(epoch, accu_loss.item() / (step + 1))
        #print('futime:{}'.format(futime))
        #print('fustat:{}'.format(fustat))
        #print(pred.cpu().tolist())

    # f.close()
    #import pdb;pdb.set_trace()
    acc = accuracy_cox(pred_all.data, fustat_torch)
    pvalue_pred = cox_log_rank(pred_all.data, fustat_torch, survtime_torch)
    c_index = CIndex_lifeline(pred_all.data, fustat_torch, survtime_torch)
    return accu_loss.item() / (step + 1), acc, pvalue_pred, c_index

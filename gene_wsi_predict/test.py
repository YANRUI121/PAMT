import os
import math
import argparse
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from regularization import Regularization
#from torchsummary import summary
#from pytorch_summary import torchsummary
from torchinfo import summary

#from my_dataset import MyDataSet
from predict_wsi_gene_dataset import MyDataSet as WSI_Gene_DataSet
# from wsi_dataset_cox import MyDataSet as WSI_Dataset
#from vit_model_gene_wsi_concat import my_model as create_model_wsi_gene
from vit_model_gene_wsi_concat_label import my_model as create_model_wsi_gene
#from vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
from temp_yr.vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
from vit_model_one_cls import my_model as create_model_wsi
#from utils_cox import read_split_data, train_one_epoch, evaluate, predict
from utils_cox_predict import read_split_data, predict
from torch.nn import DataParallel
import shutil


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tb_writer = SummaryWriter(log_dir=args.log_dir)

    val_dataset = WSI_Gene_DataSet(args.wsi_valid_feat_dir, args.gene_valid_feat_dir, args.cox_txt_path, mode='valid')
    print('valid patient count: {}'.format(str(len(val_dataset))))

    #batch_size = int(args.batch_size / len(args.device.split(',')))
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#    nw = 0 
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=False,
                                             collate_fn=val_dataset.collate_fn)

    dpr_rate = 0.1
    if args.train_flag == 0:
        if args.contrastive_loss_flag:
            model = create_model_wsi_gene(num_classes=args.num_classes, has_logits=False, wsi_block=6, gene_block=2, dpr=dpr_rate)
        else:
            model = create_model_wsi_gene_no_contrastive_loss(num_classes=args.num_classes, has_logits=False, wsi_block=6, gene_block=2, dpr=dpr_rate)
    elif args.train_flag == 1:
        model = create_model_wsi(num_classes=args.num_classes, has_logits=False, wsi_block=2, dpr=dpr_rate)
    elif args.train_flag == 2:
        model = create_model_wsi_gene(num_classes=args.num_classes, has_logits=False, gene_block=2, dpr=dpr_rate)
    else:
        raise ValueError('Invalid train flag : {}'.format(str(args.train_flag)))
    
    shutil.copy(os.path.join(os.getcwd(), sys.argv[0]), args.log_dir)
    model_log_path = os.path.join(args.log_dir, 'model_log.txt')
    model_log = open(model_log_path, 'w')
    model_log.write(str(model))
    model_log.write('\n')
    model_log.write('Total params: {:.2f}M\n'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))  # 输出参数数量
    model.cuda()
    #model_report, _ = torchsummary.summary_string(model, [(500, 1280), (186,5245)], device='cuda') 
    if args.train_flag == 0:
        model_log.write(str(summary(model, [(500, 384), (186,5245)], device='cuda')))
    elif args.train_flag == 1:
        model_log.write(str(summary(model, input_size=(1, 500, 2048), device='cuda')))

    model_log.close()

    
    model = DataParallel(model, device_ids=None)
    model=model.cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
        
        # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        #     else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 
        #             'head.bias', 'patch_embed.proj.bias', 'patch_embed.proj.weight']
        # for k in del_keys:
        #     del weights_dict[k]
        # torch.nn.init.kaiming_normal_(model.patch_embed.proj_conv.weight,mode='fan_out',nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(model.patch_embed.proj_lin.weight,mode='fan_out',nonlinearity='relu')

        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            #
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
               print("training {}".format(name))


    pg = [p for p in model.parameters() if p.requires_grad]
    if args.weight_decay > 0:
        regular_loss = Regularization(model, args.weight_decay, p=1).to(device)
    else:
        regular_loss = False


    criterion = torch.nn.CrossEntropyLoss().cuda()
    val_loss, val_cox_acc, val_p_value, val_c_index = predict(model=model,
                                     topK=args.topK,
                                     criterion=criterion,
                                     data_loader=val_loader,
                                     json_path='valid_log.txt',
                                     save_attn_dir=args.save_attn_dir,
                                     reg_loss=regular_loss,
                                     train_flag = args.train_flag,
                                     contrastive_loss_flag=args.contrastive_loss_flag)
    print('val_loss:{}, val_cox_acc:{}, val_p_value:{}, val_c_index:{}'.format(str(val_loss), str(val_cox_acc), str(val_p_value), str(val_c_index)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--topK', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0)    #正则化参数
    parser.add_argument('--log_dir', type=str, default='log_lusc_norm/test',
                        help='log directory')
    parser.add_argument('--train_flag', type=int, default=0,
                        help='train mode, 0: wsi + gene, 1: wsi, 2: gene')
    parser.add_argument('--contrastive_loss_flag', type=int, default=0,
                        help='train mode, 0: no contrastive loss, 1: contrastive loss')
    #parser.add_argument('--cox_txt_path', type=str, default='luad_gene_data/cox_time_luad.txt')
    parser.add_argument('--cox_txt_path', type=str, default="luad_gene_data/cox_time_luad.txt")
    parser.add_argument('--wsi_train_feat_dir', type=str,
                        default='/home/sdd/zxy/TCGA_data/luad_whole_slide_select_feat_txt_dino/train')
    parser.add_argument('--wsi_valid_feat_dir', type=str,
                        default='/home/sdd/zxy/TCGA_data/luad_whole_slide_select_feat_txt_dino/train')
    parser.add_argument('--gene_train_feat_dir', type=str,
                        default='luad_gene_data/pathway_gene_features')
    parser.add_argument('--gene_valid_feat_dir', type=str,
                        default='luad_gene_data/pathway_gene_features')
   # parser.add_argument('--gene_train_feat_dir', type=str,
   #                     default='luad_gene_data/pathway_gene_features')
   # parser.add_argument('--gene_valid_feat_dir', type=str,
   #                     default='luad_gene_data/pathway_gene_features')
    parser.add_argument('--save_attn_dir', type=str,
                        default='/home/sda/zxy/luad_whole_slide_attn_single_txt/train_80_80_80')
    parser.add_argument('--weights', type=str, default='temp_yr/yanrui_gene80_wsi80_cross80QKV/model-latest.pth',
                        help='initial weights path')
    #parser.add_argument('--weights', type=str, default='log/wsi_6_gene_2_adam_lr_1E-3_l2_5E-4_dpr_1E-2_noconloss_20230823/model-best.pth',
    #                    help='initial weights path')
    #parser.add_argument('--weights', type=str, default='log_luad_100dino/wsi_6_gene_2_adam_lr_1E-3_l2_1E-4_dpr_1E-1_bs128_whole_20231113/model-sum-0.5798.pth',
    #                    help='initial weights path')
    parser.add_argument( '--freeze-layers', type=bool, default=False )
    parser.add_argument('--device', default='3,4,5', type=str, help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)

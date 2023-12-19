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
from wsi_gene_dataset import MyDataSet as WSI_Gene_DataSet
# from wsi_dataset_cox import MyDataSet as WSI_Dataset
#from vit_model_gene_wsi_concat import my_model as create_model_wsi_gene
from vit_model_gene_wsi_concat_label import my_model as create_model_wsi_gene
from temp_yr.vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
#from vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
from vit_model_one_cls import my_model as create_model_wsi
from utils_cox import read_split_data, train_one_epoch, evaluate 
from torch.nn import DataParallel
import shutil


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tb_writer = SummaryWriter(log_dir=args.log_dir)


    train_dataset = WSI_Gene_DataSet(args.wsi_train_feat_dir, args.gene_train_feat_dir, args.cox_txt_path,  mode='train')
    print('train patient count: {}'.format(str(len(train_dataset))))
    val_dataset = WSI_Gene_DataSet(args.wsi_valid_feat_dir, args.gene_valid_feat_dir, args.cox_txt_path, mode='valid')
    print('valid patient count: {}'.format(str(len(val_dataset))))


    #batch_size = int(args.batch_size / len(args.device.split(',')))
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #nw = 0 
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=nw,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=False,
                                             collate_fn=val_dataset.collate_fn)

    model_dpr = 0.2
    wsi_block = 6
    if args.train_flag == 0:
        if args.contrastive_loss_flag:
            model = create_model_wsi_gene(num_classes=args.num_classes, has_logits=False, wsi_block=wsi_block, gene_block=2, dpr=model_dpr)
        else:
            model = create_model_wsi_gene_no_contrastive_loss(num_classes=args.num_classes, has_logits=False, wsi_block=wsi_block, gene_block=2, dpr=model_dpr)
    elif args.train_flag == 1:
        model = create_model_wsi(num_classes=args.num_classes, has_logits=False, wsi_block=wsi_block, dpr=model_dpr)
    elif args.train_flag == 2:
        model = create_model_wsi_gene(num_classes=args.num_classes, has_logits=False, gene_block=2, dpr=model_dpr)
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
    #print(model)  # 输出模型结构
    #print('Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))  # 输出参数数量
    ##  运行模型并输出显存占用情况
    #input_tensor1 = torch.randn(1, 186, 5245).cuda()
    #input_tensor2 = torch.randn(1, 500, 1280).cuda()
    #model=model.cuda()
    #with torch.no_grad():
    #    output = model(input_tensor2, input_tensor1)
    #print('GPU memory used:', torch.cuda.memory_allocated())
    
    model = DataParallel(model, device_ids=None)
    model=model.cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
        #
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 
                    'head.bias', 'patch_embed.proj.bias', 'patch_embed.proj.weight']
        for k in del_keys:
            del weights_dict[k]
        torch.nn.init.kaiming_normal_(model.patch_embed.proj_conv.weight,mode='fan_out',nonlinearity='relu')
        torch.nn.init.kaiming_normal_(model.patch_embed.proj_lin.weight,mode='fan_out',nonlinearity='relu')

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
    #optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-5)
    #optimizer = optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.999), eps=1E-3, weight_decay=5E-4)
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=1E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    #criterion = torch.nn.BCEWithLogitsLoss().cuda()

    best_val_cindex = 0
    best_sum_cindex = 0
    save_name_txt = os.path.join(args.log_dir, "train_valid_acc.txt")
    model_file = open(save_name_txt, "w")
    for epoch in range(args.epochs):
        # train
        train_loss, train_cox_acc,train_p_value, train_c_index = train_one_epoch(model=model,
                                                topK=args.topK,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                #device=device,
                                                epoch=epoch,
                                                reg_loss=regular_loss,
                                                train_flag = args.train_flag,
                                                contrastive_loss_flag=args.contrastive_loss_flag)

        scheduler.step()

        # validate
        val_loss, val_cox_acc, val_p_value, val_c_index = evaluate(model=model,
                                     topK=args.topK,
                                     criterion=criterion,
                                     data_loader=val_loader,
                                     #device=device,
                                     epoch=epoch,
                                     json_path='valid_log.txt',
                                     reg_loss=regular_loss,
                                     train_flag = args.train_flag,
                                     contrastive_loss_flag=args.contrastive_loss_flag)


        #tags = ["train_loss", "train_p_value", "val_loss", "val_acc", "learning_rate"]
               # "train_tpr", "train_tnr", "val_tpr", "val_tnr"]
        tb_writer.add_scalar('train_loss', train_loss, epoch)
        tb_writer.add_scalar('train_cox_acc', train_cox_acc, epoch)
        tb_writer.add_scalar('train_p_value', train_p_value, epoch)
        tb_writer.add_scalar('train_c_index', train_c_index, epoch)
        tb_writer.add_scalar('val_loss', val_loss, epoch)
        tb_writer.add_scalar('val_cox_acc', val_cox_acc, epoch)
        tb_writer.add_scalar('val_p_value', val_p_value, epoch)
        tb_writer.add_scalar('val_c_index', val_c_index, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
        #tb_writer.add_scalar(tags[5], train_tpr, epoch)
        #tb_writer.add_scalar(tags[6], train_tnr, epoch)
        #tb_writer.add_scalar(tags[7], val_tpr, epoch)
        #tb_writer.add_scalar(tags[8], val_tnr, epoch)
        model_file.write('Train-Epoch-' + str(epoch) + ' : train loss : ' + str(train_loss) +  ' ; train cox acc : ' + str(train_cox_acc)
                + ' ; train p value : ' + str(train_p_value) + ' ; train c index : ' + str(train_c_index) + '\n')
        model_file.write('Valid-Epoch-' + str(epoch) + ' : valid loss : ' + str(val_loss) + ' ; valid cox acc : ' + str(val_cox_acc)
                + ' ; valid p value : ' + str(val_p_value) + ' ; valid c index : ' + str(val_c_index) + '\n')
        model_file.write('lrlrl-Epoch-' + str(epoch) + ' : learning rate : ' + str(optimizer.param_groups[0]["lr"]) + '\n')
        model_file.flush()
        torch.save(model.state_dict(), os.path.join(args.log_dir,'model-latest.pth'))
        if val_c_index >= best_val_cindex:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model-val-best.pth'))
            if train_c_index >= 0.9:
                os.rename(os.path.join(args.log_dir, 'model-val-best.pth'), os.path.join(args.log_dir, 'model-val-{}.pth'.format(str(round(val_c_index, 4)))))
            best_val_cindex = val_c_index
            model_file.write('save best val c_index {} checkpoint'.format(str(val_c_index)) + '\n')
        if val_c_index + train_c_index >= best_sum_cindex:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model-sum-best.pth'))
            if train_c_index >= 0.9:
                os.rename(os.path.join(args.log_dir, 'model-sum-best.pth'), os.path.join(args.log_dir, 'model-sum-{}.pth'.format(str(round(val_c_index, 4)))))
            best_sum_cindex = val_c_index + train_c_index
            model_file.write('save best sum c_index {} checkpoint'.format(str(best_sum_cindex)) + '\n')
    model_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--topK', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0)    #正则化参数
    parser.add_argument('--log_dir', type=str, default='log_lusc_norm/wsi_6_gene_2_adam_lr_1E-3_l2_1E-5_dpr_2E-1_bs64_whole_40_40_4_20231121',
                        help='log directory')
    parser.add_argument('--train_flag', type=int, default=0,
                        help='train mode, 0: wsi + gene, 1: wsi, 2: gene')
    parser.add_argument('--contrastive_loss_flag', type=int, default=0,
                        help='train mode, 0: no contrastive loss, 1: contrastive loss')
    parser.add_argument('--cox_txt_path', type=str, default="lusc_gene_data/cox_time_lusc.txt")
    parser.add_argument('--wsi_train_feat_dir', type=str,
                        default='/home/sdd/zxy/TCGA_data/lusc_whole_select_feat_txt_dino/train')
    parser.add_argument('--wsi_valid_feat_dir', type=str,
                        default='/home/sdd/zxy/TCGA_data/lusc_whole_select_feat_txt_dino/valid')
    parser.add_argument('--gene_train_feat_dir', type=str,
                        default='lusc_gene_data/pathway_gene_features')
    parser.add_argument('--gene_valid_feat_dir', type=str,
                        default='lusc_gene_data/pathway_gene_features')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument( '--freeze-layers', type=bool, default=False )
    parser.add_argument('--device', default='7,6,5,4', type=str, help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)

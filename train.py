# -*- encoding: utf-8 -*-
# here put the import lib
import pandas as pd
import numpy as np
from utils.DataLoade import CustomDataset
from torch.utils.data import DataLoader
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
from model.DeepLab import DeepLabV3
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from utils.eval_tool import label_accuracy_score
from utils.data_txt import image2csv
import argparse
from tqdm import tqdm
import time

#   引用u3+模型
from u3plus.UNet_3Plus import UNet_3Plus
from u3plus.UNet_3Plus import UNet_3Plus_DeepSup

#   引用parser
from CommandLine.train_parser import get_args_parser


def load_model(args):
    if args.model == 'Unet':
        model_name = 'UNet'
        net = UNet(3, nb_classes)
        print("using UNet")
    elif args.model == "FCN":
        model_name = 'FCN8x'
        net = FCN8x(args.nb_classes)
        print("using FCN")
    elif args.model == "Deeplab":
        model_name = 'DeepLabV3'
        net = DeepLabV3(nb_classes)
        print("using DeeplabV3")
    elif args.model == 'Unet3+':
        model_name = 'Unet3+'
        net = UNet_3Plus()
        print("using UNet3+")
    elif args.model == 'Unet3+_Sup':
        model_name = 'Unet3+_Sup'
        net = UNet_3Plus_DeepSup()
        print("using UNet3+_Sup")

    return model_name, net


def train(args, model_name, net):
    model_path = './model_result/best_model_{}.mdl'.format(model_name)
    result_path = './result_{}.txt'.format(model_name)

    if os.path.exists(result_path):
        os.remove(result_path)

    best_score = 0.0
    start_time = time.time()  # 开始训练的时间
    # 加载模型
    net.loadIFExist(model_path)
    # 构建网络
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_csv_dir = 'train.csv'
    val_csv_dir = 'val.csv'
    train_data = CustomDataset(train_csv_dir, args.input_height, args.input_width)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_data = CustomDataset(val_csv_dir, args.input_height, args.input_width)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        torch.cuda.set_device(args.gpu)
        net.cuda()
        criterion = criterion.cuda()
    epoch = args.epochs
    for e in range(epoch):
        net.train()
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        #   train的进度条
        with tqdm(total=len(train_dataloader), desc=f'{e + 1}/{epoch} epoch Train_Progress') as pb_train:
            for i, (batchdata, batchlabel) in enumerate(train_dataloader):
                if use_gpu:
                    batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()

                output = net(batchdata)
                output = F.log_softmax(output, dim=1)
                loss = criterion(output, batchlabel)

                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = batchlabel.data.cpu()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item() * batchlabel.size(0)
                label_true = torch.cat((label_true, real), dim=0)
                label_pred = torch.cat((label_pred, pred), dim=0)
                pb_train.update(1)

        train_loss /= len(train_data)
        acc, acc_cls, mean_iu, fwavacc, _, _, _, _ = label_accuracy_score(label_true.numpy(), label_pred.numpy(),
                                                                          NUM_CLASSES)

        print(
            f'epoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f}')
        print(f'Time for this epoch: {time.time() - epoch_start_time:.2f} seconds')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f}')

        net.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        with tqdm(total=len(val_dataloader), desc=f'{e + 1}/{epoch} epoch Val_Progress') as pb_val:
            with torch.no_grad():
                for i, (batchdata, batchlabel) in enumerate(val_dataloader):
                    if use_gpu:
                        batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()

                    output = net(batchdata)
                    output = F.log_softmax(output, dim=1)
                    loss = criterion(output, batchlabel)

                    pred = output.argmax(dim=1).squeeze().data.cpu()
                    real = batchlabel.data.cpu()

                    val_loss += loss.cpu().item() * batchlabel.size(0)
                    val_label_true = torch.cat((val_label_true, real), dim=0)
                    val_label_pred = torch.cat((val_label_pred, pred), dim=0)

                    pb_val.update(1)

            val_loss /= len(val_data)
            val_acc, val_acc_cls, val_mean_iu, val_fwavacc, _, _, _, _ = label_accuracy_score(val_label_true.numpy(),
                                                                                              val_label_pred.numpy(),
                                                                                              NUM_CLASSES)

        print(
            f'epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}')

        score = (val_acc_cls + val_mean_iu) / 2
        if score > best_score:
            best_score = score
            torch.save(net.state_dict(), model_path)

    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    model_name, net = load_model(args)
    print(args.init_lr)
    # train(args,model_name,net)

import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset import TransferDataset
from VGG_with_decoder import Encoder, Decoder


def load_nets():
    vgg = models.vgg19(pretrained=True)
    net_e = Encoder(vgg.features)
    net_d = Decoder()
    return net_e, net_d


def get_dataloader(content_root):
    transferset = TransferDataset(content_root)
    loader = DataLoader(transferset, 8, True, num_workers=8, drop_last=True)
    return loader


def get_loss(encoder, decoder, content, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
    fc = encoder(content)
    content_new = decoder(*fc, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
    fc_new = encoder(content_new)
    mse_loss = nn.MSELoss()
    loss_r = mse_loss(content_new, content)
    loss_p_list = []
    for i in range(5):
        loss_p_list.append(mse_loss(fc_new[i], fc[i]))
    loss_p = sum(loss_p_list) / len(loss_p_list)
    loss = 0.5 * loss_r + 0.5 * loss_p
    return loss


def train_single_epoch(args, epoch, encoder, decoder, loader, optimizer, alpha_train=0):
    for i, content_batch in enumerate(loader):
        content_batch.requires_grad = False

        d0_control = args.d_control[:5]
        d1_control = args.d_control[5: 8]
        d2_control = args.d_control[9: 16]
        d3_control = args.d_control[16: 23]
        d4_control = args.d_control[23: 28]
        d5_control = args.d_control[28: 32]
        d0_control = [int(i) for i in d0_control]
        d1_control = [int(i) for i in d1_control]
        d2_control = [int(i) for i in d2_control]
        d3_control = [int(i) for i in d3_control]
        d4_control = [int(i) for i in d4_control]
        d5_control = [int(i) for i in d5_control]

        if args.gpu is not None:
            content_batch = content_batch.cuda()
        loss = get_loss(encoder, decoder, content_batch, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if i % 100 == 0:
            print('epoch: %d | batch: %d | loss: %.4f' % (epoch, i, loss.cpu().data))

        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()


def train(args, encoder, decoder):
    MAX_EPOCH = args.max_epoch
    content_root = args.training_dataset

    for param in encoder.parameters():
        param.requires_grad = False

    decoder.train(), encoder.eval()
    loader = get_dataloader(content_root)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for i in range(MAX_EPOCH):
        train_single_epoch(args, i, encoder, decoder, loader, optimizer)
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, '{:s}/decoder_epoch_{:d}.pth.tar'.format(args.save_dir, i + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0)
    parser.add_argument('-s', '--save_dir', default='trained_models')
    parser.add_argument('-d', '--d_control', default='01010000000100000000000000001111')
    parser.add_argument('-me', '--max_epoch', default=2, type=int)
    parser.add_argument('-t', '--train_data')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net_e, net_d = load_nets()

    if args.gpu is not None:
        net_e.cuda()
        net_d.cuda()

    train(args, net_e, net_d)
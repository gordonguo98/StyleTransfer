import argparse
import os
import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import TransferDataset
from VGG_with_decoder import Encoder, Decoder


def load_nets():
    vgg = torch.load('./vgg_normalised_conv5_1.pth')
    net_e = Encoder(vgg)
    net_d = Decoder()
    return net_e, net_d


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


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.5 ** epoch)


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
            print('%s epoch: %d | batch: %d | loss: %.4f' % (datetime.datetime.now(), epoch, i, loss.cpu().data))

        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()


def train(args, encoder, decoder):
    MAX_EPOCH = args.max_epoch
    content_root = args.train_data

    for param in encoder.parameters():
        param.requires_grad = False

    decoder.train(), encoder.eval()
    loader = DataLoader(TransferDataset(content_root), args.batch_size, True, num_workers=args.num_workers, drop_last=True)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    for i in range(MAX_EPOCH):
        train_single_epoch(args, i, encoder, decoder, loader, optimizer)
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, '{:s}/decoder_epoch_{:d}.pth'.format(args.save_dir, i + 1))
        adjust_learning_rate(optimizer, i + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0)
    parser.add_argument('-s', '--save_dir', default='trained_models')
    # 01010000000100000000000000001111
    parser.add_argument('-d', '--d_control', default='01010000000100000000000000001111')
    parser.add_argument('-me', '--max_epoch', default=5, type=int)
    parser.add_argument('-t', '--train_data')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
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
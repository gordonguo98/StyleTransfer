import sys
import os

import numpy as np

import cv2
import torch

import argparse
from torchvision import utils
from torchvision import transforms
from VGG_with_decoder import Encoder, Decoder0, Decoder1, Decoder2, Decoder3, Decoder4, Decoder5
from wct import transform

abs_dir = os.path.abspath(os.path.dirname(__file__))


def load_net(args):
    encoder_param = torch.load('./vgg_normalised_conv5_1.pth')
    model_path = os.path.join('./trained_models', args.trained_model)
    net_e = Encoder(encoder_param)
    net_d0 = Decoder0()
    net_d0.load_state_dict(torch.load(model_path))
    net_d1 = Decoder1()
    net_d1.load_state_dict(torch.load(model_path))
    net_d2 = Decoder2()
    net_d2.load_state_dict(torch.load(model_path))
    net_d3 = Decoder3()
    net_d3.load_state_dict(torch.load(model_path))
    net_d4 = Decoder4()
    net_d4.load_state_dict(torch.load(model_path))
    net_d5 = Decoder5()
    net_d5.load_state_dict(torch.load(model_path))
    return net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5


def get_a_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def resize_save(content, style, out):
    if content.shape[0] < content.shape[1]:
        out_h = 512
        out_w = np.int32(512.0 * content.shape[1] / content.shape[0])
    else:
        out_w = 512
        out_h = np.int32(512.0 * content.shape[0] / content.shape[1])
    content = cv2.resize(content, (out_w, out_h), cv2.INTER_AREA)
    style = cv2.resize(style, (out_w, out_h), cv2.INTER_AREA)
    out = cv2.resize(out, (out_w, out_h), cv2.INTER_AREA)
    return content, style, out


def resize_imgs(content, style):
    c_h = 512
    c_w = 768

    content = cv2.resize(content, (c_w, c_h), cv2.INTER_AREA)
    style = cv2.resize(style, (content.shape[1], content.shape[0]), cv2.INTER_AREA)
    return content, style


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=1)
    parser.add_argument('-sd', '--save_dir', default='./result_val_nas')
    parser.add_argument('-c', '--content')
    parser.add_argument('-s', '--style')
    parser.add_argument('-a', '--alpha', default=1.0)
    parser.add_argument('-d', '--d_control', default='01010000000100000000000000001111')
    parser.add_argument('--trained_model', default='decoder_epoch_1.pth')
    args = parser.parse_args()
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5 = load_net(args)
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
        net_e.cuda(), net_e.eval()
        net_d0.cuda(), net_d0.eval()
        net_d1.cuda(), net_d1.eval()
        net_d2.cuda(), net_d2.eval()
        net_d3.cuda(), net_d3.eval()
        net_d4.cuda(), net_d4.eval()
        net_d5.cuda(), net_d5.eval()

    content_list = os.listdir(args.content)
    style_list = os.listdir(args.style)
    content_list = [i for i in content_list if '.jpg' in i]
    style_list = [i for i in style_list if '.jpg' in i]
    img_pairs = set(content_list) & set(style_list)

    for img_pair in img_pairs:
        content_path = os.path.join(args.content, img_pair)
        style_path = os.path.join(args.style, img_pair)
        print('----- transferring image %s -------' % (img_pair))
        content = get_a_image(content_path)
        style = get_a_image(style_path)
        content_save = content
        style_save = style
        content, style = resize_imgs(content, style)
        content = transforms.ToTensor()(content)
        style = transforms.ToTensor()(style)
        content = content.unsqueeze(0)
        style = style.unsqueeze(0)
        if args.gpu is not None:
            content = content.cuda()
            style = style.cuda()
        cF = list(net_e(content))
        sF = list(net_e(style))
        csF = []
        for ii in range(len(cF)):
            if ii == 0:
                if d0_control[0] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 1:
                if d2_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 2:
                if d3_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 3:
                if d4_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            elif ii == 4:
                if d5_control[-1] == 1:
                    this_csF = transform(cF[ii], sF[ii], args.alpha)
                    csF.append(this_csF)
                else:
                    csF.append(cF[ii])
            else:
                csF.append(cF[ii])

        csF[0] = net_d0(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d0(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d1_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d1(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d1(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d2_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d2(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d2(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d3_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d3(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d3(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d4_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d4(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d4(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        if d5_control[0] == 1:
            csF[0] = transform(csF[0], sF[0], args.alpha)
        csF[0] = net_d5(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        sF[0] = net_d5(*sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
        out = csF[0].cpu().data.float()
        utils.save_image(out, os.path.join(args.save_dir, img_pair))
        out = cv2.imread(os.path.join(args.save_dir, img_pair))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        content_save, style_save, out = resize_save(content_save, style_save, out)
        out_compare = np.concatenate((content_save, style_save, out), 1)
        cv2.imwrite(os.path.join(args.save_dir, img_pair), out)
        cv2.imwrite(os.path.join(args.save_dir, 'compare_' + img_pair), out_compare)
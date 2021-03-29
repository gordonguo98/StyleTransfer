import numpy as np
import torch

from PIL import Image


def whiten_and_color(cF, sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF, 1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    c_gram = torch.mm(cF,cF.t()).div(cFSize[1]-1)
    c_choose = torch.eye(cFSize[0]).double().cuda()

    contentConv = (1 - c_choose) * c_gram + 2.0 * c_choose * c_gram + 0.5 * c_gram
    c_u, c_e, c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.000001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1) + 0.0 * torch.eye(sFSize[0]).double().cuda()
    s_u, s_e, s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.000001:
            k_s = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature


def wct_segment(cF, sF, cS, sS, label_set, label_indicator):
    def resize(feat, target):
        size = (target.size(2), target.size(1))
        if len(feat.shape) == 2:
            return np.asarray(Image.fromarray(feat).resize(size, Image.NEAREST))
        else:
            return np.asarray(Image.fromarray(feat, mode='RGB').resize(size, Image.NEAREST))

    def get_index(feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0]).cuda()

    resized_content_segment = resize(cS, cF)
    resized_style_segment = resize(sS, sF)

    target_feature = cF.clone()
    for label in label_set:
        if not label_indicator[label]:
            continue
        content_index = get_index(resized_content_segment, label)
        style_index = get_index(resized_style_segment, label)
        if content_index is None or style_index is None:
            continue
        masked_content_feat = torch.index_select(cF, 1, content_index)
        masked_style_feat = torch.index_select(sF, 1, style_index)
        _target_feature = whiten_and_color(masked_content_feat, masked_style_feat)
        if torch.__version__ >= '0.4.0':
            # XXX reported bug in the original repository
            new_target_feature = torch.transpose(target_feature, 1, 0)
            new_target_feature.index_copy_(0, content_index,
                                           torch.transpose(_target_feature, 1, 0))
            target_feature = torch.transpose(new_target_feature, 1, 0)
        else:
            target_feature.index_copy_(1, content_index, _target_feature)
    return target_feature


def transform(cF, sF, cS, sS, label_set, label_indicator, alpha):
    cF = cF.double()
    sF = sF.double()
    if len(cF.size()) == 4:
        cF = cF[0]
    if len(sF.size()) == 4:
        sF = sF[0]
    C, W, H = cF.size(0),cF.size(1),cF.size(2)
    _, W1, H1 = sF.size(0),sF.size(1),sF.size(2)
    cFView = cF.view(C,-1)
    sFView = sF.view(C,-1)

    targetFeature = wct_segment(cFView, sFView, cS, sS, label_set, label_indicator)
    targetFeature = targetFeature.view_as(cF)
    alpha = int(alpha)
    csF = alpha * targetFeature + (1.0 - alpha) * cF
    csF = csF.float().unsqueeze(0)
    return csF

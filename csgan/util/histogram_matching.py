import numpy as np
import torch


def cal_hist(vec):
    hist = torch.histc(vec, bins=256, min=0, max=256)
    return (hist / hist.sum()).cumsum(dim=0)


def cal_trans(dst, ref):
    table = torch.zeros(256)
    i_dst = 1
    for i_ref in range(1, 256):
        while dst[i_dst] < ref[i_ref]:
            table[i_dst] = i_ref
            i_dst += 1
            if i_dst == 256:
                break
        if i_dst == 256:
            break
    table[i_dst:] = 255
    return table


def histogram_matching(vec_dst, vec_ref):
    vec_match = vec_dst.clone()
    for i in range(vec_dst.size(0)):
        align_dst = vec_dst[i]
        align_ref = vec_ref[i]
        hist_dst = cal_hist(align_dst)
        hist_ref = cal_hist(align_ref)
        table = cal_trans(hist_dst, hist_ref)
        vec_match[i, :] = torch.gather(table, 0, align_dst.type(torch.long))
    return vec_match

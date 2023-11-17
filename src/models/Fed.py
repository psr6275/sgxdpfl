#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg

def FedWeightAvgCDP(w, size, sensitivity, dp_mechanism, noise_scale, device):
    w_avg = FedWeightAvg(w, size)
    w_avg_dp = copy.deepcopy(w_avg)
    if dp_mechanism == 'Laplace':
        for k, v in w_avg.items():
            w_avg_dp[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * noise_scale,
                                                                size=v.shape)).to(device)
    elif dp_mechanism == 'Gaussian':
        for k, v in w_avg.items():
            w_avg_dp[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * noise_scale,
                                                                size=v.shape)).to(device)
    elif dp_mechanism == 'MA':
        # sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
        for k, v in w_avg.items():
            w_avg_dp[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * noise_scale,
                                            size=v.shape)).to(device)                                      
    return w_avg_dp
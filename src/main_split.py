#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import load_dataset_info
from utils.options import args_parser
from models.UpdateNew import LocalUpdateDP, LocalUpdateDPSerial, LocalUpdateCDP
from models.Nets import build_model
from models.Fed import FedAvg, FedWeightAvg, FedWeightAvgCDP
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, calculate_noise_scale_cdp
from utils.gradient_utils import GradSampleModule
# from opacus.grad_sample import GradSampleModule

if __name__ == '__main__':
    args = args_parser()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    data_info = load_dataset_info(args, '../data/examples')

    dataset_test = data_info['dataset_test']
    img_size = dataset_test[0][0].shape
    args.num_users = data_info['num_users']
    dict_users = data_info['dict_users']

    net_glob = build_model(args, img_size)

    # use opacus to wrap model to clip per sample gradient
    if args.dp_mechanism != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    ## Construct local models!
    # We have splitted dataset in data_info['sdatadir']
    # We can construct local class using splitted dataset
    # This part will be reflected when we seperate the client's part!
    clients = []
    for ui in range(args.num_users):
        sdata = torch.load(os.path.join(data_info['sdatadir'], "dataset_id_%s.pth"%ui))
        if args.dp_method == 'cdp':
            clients.append(LocalUpdateCDP(args, sdata))
        elif args.serial:
            clients.append(LocalUpdateDPSerial(args, sdata))
        else:
            clients.append(LocalUpdateDP(args,sdata))
    
    ## Initialize Global Weights
    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    ## Training 
    acc_test = []
    # However, we will consider frac = 1 for cross-silo setting!
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)
    
    for iter in range(args.epochs):
        t_start = time.time()
        w_locals, loss_locals, weight_locols = [], [], []
        # round-robin selection (in our basic scenario, we will ignore client selection!!)
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]        
        for idx in idxs_users:
            local = clients[idx]            
            net_local = GradSampleModule(build_model(args, img_size))
            net_local.load_state_dict(copy.deepcopy(net_glob.state_dict()))
            # this deep copy part will require communication from server to clients (for broadcasting)!
            # w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w, loss = local.train(net=net_local.to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))
            del net_local
            torch.cuda.empty_cache()

        # update global weights
        if args.dp_method != 'cdp':
            w_glob = FedWeightAvg(w_locals, weight_locols)
        else:
            if args.dp_mechanism in ['Laplace', 'Gaussian']:
                sensitivity = cal_sensitivity(args.lr,args.dp_clip, data_info['num_train'])
            elif args.dp_mechanism == 'MA':
                sensitivity = cal_sensitivity_MA(args.lr, args.dp_clip, data_info['num_train'])
            noise_scale = calculate_noise_scale_cdp(args, args.epochs)
            w_glob = FedWeightAvgCDP(w_locals, weight_locols, sensitivity, args.dp_mechanism, noise_scale, args.device)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        t_end = time.time()
        print("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        
    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid_{}_users_{}_dp_{}_{}_epsilon_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,args.num_users, args.dp_method,
                          args.dp_mechanism, args.dp_epsilon), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid_{}_users_{}_dp_{}_{}_epsilon_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.num_users, args.dp_method, args.dp_mechanism, args.dp_epsilon))

#-*-coding:utf-8-*-

import sys
import os

import yaml

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torchvision

import cv2

import copy

import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num

__ROOT_PATH = os.path.abspath('../../')
sys.path.append(__ROOT_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.rcParams['figure.facecolor'] = 'w'

from csgan.loader.colored_mnist_loader import ColoredMnistDataset
from csgan.model.cs_glow_model_huk import CSGlowMnistModel

#DATA_ROOT = os.path.join(__ROOT_PATH, "data/")
DATA_ROOT = '/data'
RUN_ROOT = os.path.join(__ROOT_PATH, "runs/")

now = time.localtime()
run_dir = os.path.join(RUN_ROOT, f"cs_glow_{now.tm_hour}_{now.tm_min}/")
os.makedirs(run_dir, exist_ok=True)

params = {'seed': 2222, 'num_epoch': 200, 'batch_size': 16, 'test_batch_size': 512,
        'learning_rate': 0.0001, 'beta1': 0.5, 'beta2': 0.999,
        'scheduler_gamma': 1., 'weight_decay': 0., #0.00001,
        'n_bits': 5,
        'lambda_identity': 1., 'lambda_glow': 1.,
        'lambda_weight_cycle': 0., 'lambda_siamese': 0.,
        'lambda_content': 0.5, 'lambda_style': 0.5,
        'gamma_content': 1., 'gamma_style': 1.,
        'scheduler_interval': 1000, 'checkpoint_interval': 10,
        'validation_interval': 10, 'logging_interval': 10, 'sampling_interval': 100,
        'stopping_loss': 'loss',
        'run_dir': run_dir}

with open (os.path.join(run_dir,'params.yaml'), 'w') as outfile:
    yaml.dump(params, outfile, default_flow_style=False)

train_dataset = ColoredMnistDataset(root=DATA_ROOT, dirname='colored_mnist_bg', train=True)
test_dataset = ColoredMnistDataset(root=DATA_ROOT, dirname='colored_mnist_bg', train=False)
print(len(train_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, params['batch_size'], shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, params['test_batch_size'], shuffle=False, drop_last=False)
print(len(train_loader), len(test_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cs_model = CSGlowMnistModel(device)
#cs_model.load(os.path.join(run_dir, 'best_model.pth.tar'))
cs_model.train_model(train_loader, params=params)

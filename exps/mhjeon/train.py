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

matplotlib.rcParams['figure.facecolor'] = 'w'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from csgan.loader.beautygan_loader import BeatyganDataset
from csgan.model.cs_double_encoder_model import CSDoubleEconderBeautyganModel

#DATA_ROOT = os.path.join(__ROOT_PATH, "data/")
DATA_ROOT = "/data/"
RUN_ROOT = os.path.join(__ROOT_PATH, "runs/")

now = time.localtime()
run_dir = os.path.join(RUN_ROOT, f"beautygan_sc_separate_{now.tm_hour}_{now.tm_min}/")
os.makedirs(run_dir, exist_ok=True)

params = {'seed': 2222, 'num_epoch': 200, 'batch_size': 8, 'test_batch_size': 512,
           'learning_rate': 0.0001, 'scheduler_gamma': 0.98, 'weight_decay': 0.00001,# 'clip_size': 3.,
           'lambda_identity': 5., 'lambda_cycle': 5., 'lambda_weight_cycle': 1., 'lambda_siamese': 1.,
           'lambda_content': 1., 'lambda_style': 1., 'lambda_content_seg': 0.1, 'lambda_style_seg': 0.1, 'lambda_source': 0.01, 'lambda_reference': 0.01,
           'gamma_content': 1., 'gamma_style': 1., 'gamma_content_seg': 1., 'gamma_style_seg': 1., 'gamma_source': 1., 'gamma_reference': 1.,
           'scheduler_interval': 1000, 'checkpoint_interval': 10,
           'validation_interval': 10, 'logging_interval': 10, 'sampling_interval': 1000,
           'stopping_loss': 'loss',
           'beta1': 0.9, 'beta2': 0.99,
           'run_dir': run_dir}

with open (os.path.join(run_dir,'params.yaml'), 'w') as outfile:
    yaml.dump(params, outfile, default_flow_style=False)

train_dataset = BeautyganDataset(root=DATA_ROOT, image_size=256, seg_size=64)
print(len(train_dataset))

train_loader = DataLoader(train_dataset, params['batch_size'], shuffle=True, drop_last=True)
print(len(train_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sc_model = CSDoubleEconderBeautyganModel(device)
#sc_model.load(os.path.join(run_dir, 'best_model.pth.tar'))
sc_model.train_model(train_loader, params=params)

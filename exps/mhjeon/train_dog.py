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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from csgan.loader.style_dog_loader import StylingDogDataset
from csgan.model.cs_double_encoder_model import CSDoubleEnconderStylingDogModel

#DATA_ROOT = os.path.join(__ROOT_PATH, "data/")
DATA_ROOT = "/data/"
RUN_ROOT = os.path.join(__ROOT_PATH, "runs/")

now = time.localtime()
run_dir = os.path.join(RUN_ROOT, f"stylingdog_sc_separate_{now.tm_hour}_{now.tm_min}/")
os.makedirs(run_dir, exist_ok=True)

params = {'seed': 2222, 'num_epoch': 200, 'batch_size': 8, 'test_batch_size': 512,
        'learning_rate': 0.0002, 'beta1': 0.5, 'beta2': 0.999,
        'scheduler_gamma': 1., 'weight_decay': 0., #0.00001,
        'lambda_identity': 5., 'lambda_cycle': 10., 'lambda_content': 0.01, 'lambda_style': 0.,
        'lambda_source': 0.1, 'lambda_reference': 0.1, 'lambda_content_seg': 0., 'lambda_style_seg': 0.,
        'lambda_compatible': 0.01, 'lambda_siamese': 0.,
        'gamma_content': 1., 'gamma_style': 1., 'gamma_source': 1., 'gamma_reference': 1.,
        'gamma_content_seg': 0., 'gamma_style_seg': 0.,  'gamma_compatible': 1.,
        'scheduler_interval': 1000, 'checkpoint_interval': 10,
        'validation_interval': 10, 'logging_interval': 10, 'sampling_interval': 100,
        'stopping_loss': 'loss',
        'run_dir': run_dir}

with open (os.path.join(run_dir,'params.yaml'), 'w') as outfile:
    yaml.dump(params, outfile, default_flow_style=False)

train_dataset = StylingDogDataset(root=DATA_ROOT, image_size=256)
print(f"train data set: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, params['batch_size'], shuffle=True, drop_last=True)
print(f"train loader: {len(train_loader)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sc_model = CSDoubleEnconderStylingDogModel(device)
#sc_model.load(os.path.join(run_dir, 'best_model.pth.tar'))
sc_model.train_model(train_loader, params=params)

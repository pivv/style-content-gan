import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

from abc import abstractmethod

import time

from collections import defaultdict

import numpy as np

import cv2

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter


class BaseModel(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self._device = device
        self._optimizers = List[Tuple[Any, str]]
        self._schedulers = List[Any]

    def train_on_batch(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                       global_step: int = 0) -> Dict[str, Any]:
        self.train()

        output: Dict[str, Tensor] = self.forward(batch, global_step=global_step)
        loss_dict: Dict[str, Tensor] = self.loss_function(batch, output, global_step=global_step)

        self._update_optimizers(loss_dict, params, global_step=global_step)
        for optimizer, loss_str in self._optimizers:
            if loss_str in loss_dict:
                optimizer.zero_grad()
                loss_dict[loss_str].backward()
                if 'clip_size' in params:
                    clip_grad_norm_(self.parameters(), params['clip_size'])
                optimizer.step()

        loss_dict: Dict[str, Any] = {key: value.item() for key, value in loss_dict.items()}
        return loss_dict

    def evaluate(self, data: DataLoader, global_step: int = 0) -> Dict[str, Any]:
        loss_sum, correct_sum, total_count = 0., 0, 0
        batches: Dict[str, List[Tensor]] = defaultdict(list)
        outputs: Dict[str, List[Tensor]] = defaultdict(list)
        for batch in data:
            batch: Dict[str, Tensor] = {key: value.to(self._device) for key, value in batch.items()}
            with torch.no_grad():
                output: Dict[str, Tensor] = self.forward(batch, global_step=global_step)
            for key in batch.keys():
                batches[key].append(batch[key].detach().cpu().numpy())
            for key in output.keys():
                outputs[key].append(output[key].detach().cpu().numpy())
        all_batch: Dict[str, Tensor] = {key: torch.cat(value, dim=0) for key, value in batches.items()}
        all_output: Dict[str, Tensor] = {key: torch.cat(value, dim=0) for key, value in outputs.items()}
        loss_dict: Dict[str, Tensor] = self.loss_function(all_batch, all_output, global_step=global_step)

        loss_dict: Dict[str, Any] = {key: value.item() for key, value in loss_dict.items()}
        return loss_dict

    def train_model(self, train_data: DataLoader, val_data: DataLoader = None, params: Dict[str, Any] = None,
                    optimizer=None, scheduler=None) -> None:
        torch.cuda.empty_cache()
        np.random.seed(params['seed'])

        run_dir = params['run_dir']
        checkpoint_dir = os.path.join(run_dir, 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_dir = os.path.join(run_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

        self.set_optimizers(params)

        start_time = time.time()
        istep, best_istep, best_val_loss = 0, -1, None
        num_epoch = params['num_epoch']
        loss_dict: dict = None
        val_loss_dict: dict = None
        for iepoch in range(1, num_epoch+1):
            for batch in train_data:
                batch: Dict[str, Tensor] = {key: value.to(self._device) for key, value in batch.items()}
                loss_dict: Dict[str, Any] = self.train_on_batch(batch, params, global_step=istep)
                istep += 1
                for key in loss_dict:
                    writer.add_scalar(f'train/{key}', loss_dict[key], global_step=istep)
                if istep % params['checkpoint_interval'] == 0:
                    torch.save(self.state_dict(), os.path.join(checkpoint_dir, f'model_{istep}.pth.tar'))
                if val_data is not None and istep % params['validation_interval'] == 0:
                    val_loss_dict: Dict[str, Any] = self.evaluate(val_data, global_step=istep)
                    for key in val_loss_dict:
                        writer.add_scalar(f'val/{key}', val_loss_dict[key], global_step=istep)
                    stopping_loss_str: str = params['stopping_loss'] if 'stopping_loss' in params else ''
                    if (best_val_loss is None or (not stopping_loss_str) or
                            val_loss_dict[stopping_loss_str] < best_val_loss):
                        best_istep, best_val_loss = istep, val_loss_dict[stopping_loss_str]
                        torch.save(self.state_dict(), os.path.join(run_dir, 'best_model.pth.tar'))
                if istep % params['logging_interval'] == 0:
                    print(f"[{iepoch}/{num_epoch}] {istep}'th step. " +
                          ". ".join(f"[{key.upper()}] {value:.6f}" for key, value in loss_dict.items()))
                    if val_data is not None:
                        print("    [VAL] " + ". ".join(f"[{key.upper()}] {value:.6f}" for key, value in
                                                       val_loss_dict.items()))
                    print(f'    Best Step: {best_istep:6d}. Elapsed Time: {time.time()-start_time:3f} seconds.')
            for scheduler in self._schedulers:
                scheduler.step()
            torch.cuda.empty_cache()
        self.load(os.path.join(run_dir, 'best_model.pth.tar'))

    def load(self, load_path: str) -> None:
        self.load_state_dict(torch.load(load_path))

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        for optimizer, loss_str in self._optimizers:
            if loss_str in loss_dict:
                optimizer.zero_grad()
                loss_dict[loss_str].backward()
                if 'clip_size' in params:
                    clip_grad_norm_(self.parameters(), params['clip_size'])
                optimizer.step()

    @abstractmethod
    def set_optimizers(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor], global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], global_step: int = 0,
                      **kwargs) -> Dict[str, Tensor]:
        pass

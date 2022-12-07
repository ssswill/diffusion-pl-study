
import copy
import inspect
import os
import traceback

import torch
import numpy as np
import importlib

from PIL import Image
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
import torch

from data import common


class EMA:
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, **kwargs):
        super().__init__()
        self.model = None
        self.loss_function = None

        self.kwargs = kwargs

        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.model.set_loss(self.loss_function)
        self.model.set_new_noise_schedule()

        # EMA
        if self.hparams.ema_scheduler:
            self.model_EMA = copy.deepcopy(self.model)
            self.EMA = EMA(beta=self.hparams.ema_decay)
        else:
            self.ema_scheduler = None

        # finish
        print('模型构建完成')

    # def forward(self, lr_hsi, hr_rgb):
    #     return self.model(lr_hsi, hr_rgb)

    def training_step(self, batch, batch_idx):

        data = batch

        loss = self.model(data.get('y_gt'), data.get('x_input'), mask=data.get('mask'))
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.hparams.ema_scheduler:
            if batch_idx > self.hparams.ema_start:
                self.EMA.update_model_average(self.model_EMA, self.model)
        if batch_idx + 1 % 10 == 0:
            torch.save(self.model.model_unet.state_dict(), self.hparams.save_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        if self.hparams.dataset in ['inpaint_dataset_mosaic']:
            output, visuals = self.model.restoration(data.get('x_input'), y_t=data.get('x_input'),
                                                     y_ground_truth=data.get('y_gt'), mask=data.get('mask'),
                                                     sample_num=self.hparams.sample_num)
        else:
            output, visuals = self.model.restoration(data.get('x_input'), sample_num=self.hparams.sample_num)

        loss = nn.L1Loss()
        mae = loss(output, data.get('y_gt'))
        common.save_image_tensor(visuals, self.hparams.save_dir + str(batch_idx) + '.png')
        common.save_image_tensor(output, self.hparams.save_dir + str(batch_idx) + '_predict.png')
        common.save_image_tensor(data.get('y_gt'), self.hparams.save_dir + str(batch_idx) + '_target.png')
        common.save_image_tensor(data.get('x_input'), self.hparams.save_dir + str(batch_idx) + '_input.png')
        self.log('mae', mae, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

        torch.save(self.model.model_unet.state_dict(), self.hparams.save_dict)
        # self.print(self.get_progress_bar_dict())

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
            self.model = self.instancialize(Model)

        except Exception as e:
            print(e)
            traceback.print_exc()

    def instancialize(self, Model):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(self.kwargs)
        return Model(**args1)

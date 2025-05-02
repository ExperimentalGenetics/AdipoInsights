import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import torch.optim as optim

import pytorch_lightning as pl

import segmentation_models_pytorch as smp
from monai.losses import DiceLoss

from typing import List
import matplotlib.pyplot as plt
from monai.metrics import compute_meandice

class HistoSegx5(pl.LightningModule):
    """
    A simple wrapper for a Unet built using segmentation_models_pytorch.
    """
    def __init__(self, model_kwargs):
        """
        model_kwargs: dict with keys 'encoder', 'classes', 'lr', 'encoder_lr', 'freeze_bn', 'freeze_bn_affine'
        """
        super().__init__()
        self.save_hyperparameters(model_kwargs)
        hparams = self.hparams
        
        # The hparams dictionary should be passed when creating the instance of the class.
        if hparams is None:
            raise ValueError("hparams must be provided.")

        # Build the model using the provided encoder and number of classes
        self.model = smp.Unet(
            encoder_name=hparams.encoder,
            encoder_weights="imagenet",
            classes=hparams.classes,
        )
        self.freeze_bn = hparams.get("freeze_bn", True)
        self.freeze_bn_affine = hparams.get("freeze_bn_affine", True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure optimizer: use a different learning rate for the encoder if specified.
        """
        hparams = self.hparams
        if "encoder_lr" in hparams:
            print(f"Using encoder lr {hparams['encoder_lr']} and {hparams['lr']} lr for the rest.")
            opt = optim.Adam([
                {'params': self.model.decoder.parameters()},
                {'params': self.model.segmentation_head.parameters()},
                {'params': self.model.encoder.parameters(), 'lr': hparams["encoder_lr"]}
            ], lr=hparams["lr"])
        else:
            print(f"Using {hparams['lr']} for all parameters.")
            opt = optim.Adam(self.model.parameters(), lr=hparams['lr'])
        return opt


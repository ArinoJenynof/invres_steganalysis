# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:39:23 2024

@author: Arino Jenynof
"""
import torch
from torchmetrics.classification import Accuracy
import lightning as L
from .preprocessing_convolution import PreProcessingConv2d
from .inverted_residual import InvertedResidual
from .spatial_pyramid_pooling import SpatialPyramidPool2d


class InvResModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv_layers = torch.nn.Sequential(
			PreProcessingConv2d(),
			InvertedResidual(30, 3, 36, 30, 1, 1, False),
			InvertedResidual(30, 3, 48, 30, 1, 1, False),
			torch.nn.AvgPool2d(2, 2, 1),
			InvertedResidual(30, 3, 48, 40, 1, 1, False),
			InvertedResidual(40, 3, 54, 40, 1, 1, False),
			torch.nn.AvgPool2d(2, 2, 1),
			InvertedResidual(40, 3, 54, 50, 1, 1, False),
			InvertedResidual(50, 3, 72, 50, 1, 1, True),
			torch.nn.AvgPool2d(2, 2, 1),
			InvertedResidual(50, 3, 72, 60, 1, 1, True),
			InvertedResidual(60, 3, 80, 60, 1, 1, True),
		)
		
		self.pool_layers = torch.nn.ModuleList([
			SpatialPyramidPool2d("avg"),
			SpatialPyramidPool2d("max")
		])
		
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(2520, 1260),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(1260, 630),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(630, 1)
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		tmp = self.conv_layers(x)
		tmp = torch.cat([
			pool(tmp) for pool in self.pool_layers
		], 1).view(x.size(0), -1)
		return self.classifier(tmp)


class LitInvResModel(L.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = InvResModel()
		self.criterion = torch.nn.BCEWithLogitsLoss()
		self.sigmoid = torch.nn.Sigmoid()
		self.train_accuracy = Accuracy(task="binary")
		self.valid_accuracy = Accuracy(task="binary")
		self.test_accuracy = Accuracy(task="binary")
	
	def training_step(self, batch, batch_idx):
		images, targets = batch
		targets.unsqueeze_(1)
		preds = self.model(images)
		loss = self.criterion(preds, targets.float())
		self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		self.train_accuracy(self.sigmoid(preds), targets)
		return loss
	
	def on_train_epoch_end(self):
		self.log("train_acc", self.train_accuracy, prog_bar=True)
	
	def validation_step(self, batch, batch_idx):
		images, targets = batch
		targets.unsqueeze_(1)
		preds = self.model(images)
		loss = self.criterion(preds, targets.float())
		self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		self.valid_accuracy(self.sigmoid(preds), targets)
	
	def on_validation_epoch_end(self):
		self.log("valid_acc", self.valid_accuracy, prog_bar=True)
	
	def test_step(self, batch, batch_idx):
		images, targets = batch
		targets.unsqueeze_(1)
		preds = self.model(images)
		loss = self.criterion(preds, targets.float())
		self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		self.test_accuracy(self.sigmoid(preds), targets)
		
	def on_test_epoch_end(self):
		self.log("test_acc", self.test_accuracy, prog_bar=True)
	
	def configure_optimizers(self):
		optim = torch.optim.AdamW(self.model.parameters(), betas=(.75, .999), weight_decay=.001)
		lr_sched = torch.optim.lr_scheduler.OneCycleLR(optim, .01, self.trainer.estimated_stepping_batches)
		return {
			"optimizer": optim,
			"lr_scheduler": {
				"scheduler": lr_sched,
				"interval": "step"
			}
		}

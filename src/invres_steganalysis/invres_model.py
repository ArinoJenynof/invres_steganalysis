# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:39:23 2024

@author: Arino Jenynof
"""
import torch
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

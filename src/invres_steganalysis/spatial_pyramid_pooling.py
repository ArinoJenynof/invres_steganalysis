# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 20:02:40 2024

@author: Arino Jenynof
"""
from typing import Literal, Optional, Union
import torch


class SpatialPyramidPool2d(torch.nn.Module):
	def __init__(
		self,
		mode: Literal["max", "avg"],
		outputs_size: Optional[list[Union[int, tuple[int, int]]]] = [1, 2, 4]
	):
		super().__init__()
		
		if mode == "max":
			pool_fn = torch.nn.AdaptiveMaxPool2d
		elif mode == "avg":
			pool_fn = torch.nn.AdaptiveAvgPool2d
		else:
			raise NotImplementedError(f"Unknown pool mode {mode}, expected 'max' or 'avg'")
		
		self.pools = torch.nn.ModuleList([pool_fn(size) for size in outputs_size])
		
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size = x.size(0)
		channels = x.size(1)
		
		out = [pool_fn(x).view(batch_size, channels, -1) for pool_fn in self.pools]
		return torch.cat(out, dim=2)

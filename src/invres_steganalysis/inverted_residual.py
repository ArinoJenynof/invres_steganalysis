# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:44:32 2024

@author: Arino Jenynof
"""
from typing import Optional
from torch import Tensor
from torch.nn import Module, Identity
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


class InvertedResidual(Module):
	def __init__(
		self,
		in_channels: int,
		kernel_size: int,
		expanded_channels: int,
		out_channels: int,
		stride: int,
		dilation: int,
		use_se: bool
	):
		super().__init__()

		self.use_residual = (stride == 1) and (in_channels == out_channels)

		# Expansion
		if expanded_channels != in_channels:
			self.expand = Conv2dNormActivation(
				in_channels=in_channels,
				out_channels=expanded_channels,
				kernel_size=1,
			)
		else:
			self.expand = Identity()

		# Depthwise + optional Squeeze-Excite
		self.depthwise = Conv2dNormActivation(
			in_channels=expanded_channels,
			out_channels=expanded_channels,
			kernel_size=kernel_size,
			groups=expanded_channels,
			stride=1 if dilation > 1 else stride,
			dilation=dilation
		)

		if use_se:
			squeeze_channels = _make_divisible(expanded_channels // 4, 8)
			self.se = SqueezeExcitation(expanded_channels, squeeze_channels)
		else:
			self.se = Identity()

		# Projection
		self.project = Conv2dNormActivation(
			in_channels=expanded_channels,
			out_channels=out_channels,
			kernel_size=1,
			activation_layer=None
		)

	def forward(self, x: Tensor) -> Tensor:
		res = self.expand(x)
		res = self.depthwise(res)
		res = self.se(res)
		res = self.project(res)
		if self.use_residual:
			res += x
		return res

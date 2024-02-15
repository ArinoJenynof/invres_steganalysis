# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:17:18 2024

@author: Arino Jenynof
"""
import math
import torch

SRM_FILTER_WEIGHTS = {
	"1st_order": torch.tensor([
		[[
			[0.0,	0.0,	0.0,],
			[0.0,	-1.0,	1.0,],
			[0.0,	0.0,	0.0],
		]],
		[[
			[0.0,	0.0,	0.0,],
			[0.0,	-1.0,	0.0,],
			[0.0,	0.0,	1.0],
		]],
		[[
			[0.0,	0.0,	0.0,],
			[0.0,	-1.0,	0.0,],
			[0.0,	1.0,	0.0],
		]],
		[[
			[0.0,	0.0,	0.0,],
			[0.0,	-1.0,	0.0,],
			[1.0,	0.0,	0.0],
		]],
		[[
			[0.0,	0.0,	0.0,],
			[1.0,	-1.0,	0.0,],
			[0.0,	0.0,	0.0],
		]],
		[[
			[1.0,	0.0,	0.0,],
			[0.0,	-1.0,	0.0,],
			[0.0,	0.0,	0.0],
		]],
		[[
			[0.0,	1.0,	0.0,],
			[0.0,	-1.0,	0.0,],
			[0.0,	0.0,	0.0],
		]],
		[[
			[0.0,	0.0,	1.0,],
			[0.0,	-1.0,	0.0,],
			[0.0,	0.0,	0.0],
		]],
	]),

	"2nd_order": torch.tensor([
		[[
			[0.0,	0.0,	0.0,],
			[1.0,	-2.0,	1.0,],
			[0.0,	0.0,	0.0],
		]],
		[[
			[1.0,	0.0,	0.0,],
			[0.0,	-2.0,	0.0,],
			[0.0,	0.0,	1.0],
		]],
		[[
			[0.0,	1.0,	0.0,],
			[0.0,	-2.0,	0.0,],
			[0.0,	1.0,	0.0],
		]],
		[[
			[0.0,	0.0,	1.0,],
			[0.0,	-2.0,	0.0,],
			[1.0,	0.0,	0.0],
		]]
	]) / 2,

	"3rd_order": torch.tensor([
		[[
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	1.0,	-3.0,	3.0,	-1.0],
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0]
		]],
		[[
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	1.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	-3.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	3.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	-1.0]
		]],
		[[
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	1.0,	0.0,	0.0],
			[0.0,	0.0,	-3.0,	0.0,	0.0],
			[0.0,	0.0,	3.0,	0.0,	0.0],
			[0.0,	0.0,	-1.0,	0.0,	0.0]
		]],
		[[
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	1.0,	0.0],
			[0.0,	0.0,	-3.0,	0.0,	0.0],
			[0.0,	3.0,	0.0,	0.0,	0.0],
			[-1.0,	0.0,	0.0,	0.0,	0.0]
		]],
		[[
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[-1.0,	3.0,	-3.0,	1.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0]
		]],
		[[
			[-1.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	3.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	-3.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	1.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0]
		]],
		[[
			[0.0,	0.0,	-1.0,	0.0,	0.0],
			[0.0,	0.0,	3.0,	0.0,	0.0],
			[0.0,	0.0,	-3.0,	0.0,	0.0],
			[0.0,	0.0,	1.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0]
		]],
		[[
			[0.0,	0.0,	0.0,	0.0,	-1.0],
			[0.0,	0.0,	0.0,	3.0,	0.0],
			[0.0,	0.0,	-3.0,	0.0,	0.0],
			[0.0,	1.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0]
		]]
	]) / 3,

	"edge_3x3": torch.tensor([
		[[
			[-1.0,	2.0,	-1.0,],
			[2.0,	-4.0,	2.0,],
			[0.0,	0.0,	0.0],
		]],
		[[
			[0.0,	2.0,	-1.0,],
			[0.0,	-4.0,	2.0,],
			[0.0,	2.0,	-1.0],
		]],
		[[
			[0.0,	0.0,	0.0,],
			[2.0,	-4.0,	2.0,],
			[-1.0,	2.0,	-1.0],
		]],
		[[
			[-1.0,	2.0,	0.0,],
			[2.0,	-4.0,	0.0,],
			[-1.0,	2.0,	0.0],
		]]
	]) / 4,

	"edge_5x5": torch.tensor([
		[[
			[-1.0,	2.0,	-2.0,	2.0,	-1.0],
			[2.0,	-6.0,	8.0,	-6.0,	2.0],
			[-2.0,	8.0,	-12.0,	8.0,	-2.0],
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0]
		]],
		[[
			[0.0,	0.0,	-2.0,	2.0,	-1.0],
			[0.0,	0.0,	8.0,	-6.0,	2.0],
			[0.0,	0.0,	-12.0,	8.0,	-2.0],
			[0.0,	0.0,	8.0,	-6.0,	2.0],
			[0.0,	0.0,	-2.0,	2.0,	-1.0]
		]],
		[[
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[0.0,	0.0,	0.0,	0.0,	0.0],
			[-2.0,	8.0,	-12.0,	8.0,	-2.0],
			[2.0,	-6.0,	8.0,	-6.0,	2.0],
			[-1.0,	2.0,	-2.0,	2.0,	-1.0]
		]],
		[[
			[-1.0,	2.0,	-2.0,	0.0,	0.0],
			[2.0,	-6.0,	8.0,	0.0,	0.0],
			[-2.0,	8.0,	-12.0,	0.0,	0.0],
			[2.0,	-6.0,	8.0,	0.0,	0.0],
			[-1.0,	2.0,	-2.0,	0.0,	0.0]
		]]
	]) / 12,

	"square_3x3": torch.tensor([
		[[
			[-1.0,	2.0,	-1.0,],
			[2.0,	-4.0,	2.0,],
			[-1.0,	2.0,	-1.0],
		]],
	]) / 4,

	"square_5x5": torch.tensor([
		[[
			[-1.0,	2.0,	-2.0,	2.0,	-1.0],
			[2.0,	-6.0,	8.0,	-6.0,	2.0],
			[-2.0,	8.0,	-12.0,	8.0,	-2.0],
			[2.0,	-6.0,	8.0,	-6.0,	2.0],
			[-1.0,	2.0,	-2.0,	2.0,	-1.0]
		]]
	]) / 12
}


def _calculate_fan_in_and_fan_out(tensor):
	dimensions = tensor.dim()
	if dimensions < 2:
		raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

	num_input_fmaps = tensor.size(1)
	num_output_fmaps = tensor.size(0)
	receptive_field_size = 1
	if tensor.dim() > 2:
		for s in tensor.shape[2:]:
			receptive_field_size *= s
	fan_in = num_input_fmaps * receptive_field_size
	fan_out = num_output_fmaps * receptive_field_size

	return fan_in, fan_out


class PreProcessingConv2d(torch.nn.Module):
	def __init__(self, trainable: bool = True):
		super().__init__()
		
		self.weights_3x3 = torch.nn.Parameter(
			torch.cat([
				SRM_FILTER_WEIGHTS["1st_order"],
				SRM_FILTER_WEIGHTS["2nd_order"],
				SRM_FILTER_WEIGHTS["edge_3x3"],
				SRM_FILTER_WEIGHTS["square_3x3"]
			]), trainable
		)
		self.weights_5x5 = torch.nn.Parameter(
			torch.cat([
				SRM_FILTER_WEIGHTS["3rd_order"],
				SRM_FILTER_WEIGHTS["edge_5x5"],
				SRM_FILTER_WEIGHTS["square_5x5"]
			]), trainable
		)
		
		self.biases_3x3 = torch.nn.Parameter(torch.empty(self.weights_3x3.size(0)), trainable)
		self.biases_5x5 = torch.nn.Parameter(torch.empty(self.weights_5x5.size(0)), trainable)
		
		self.reset_parameters()
	
	def reset_parameters(self) -> None:
		fan_in, _ = _calculate_fan_in_and_fan_out(self.weights_3x3)
		if fan_in != 0:
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.biases_3x3, -bound, bound)
		
		fan_in, _ = _calculate_fan_in_and_fan_out(self.weights_5x5)
		if fan_in != 0:
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.biases_5x5, -bound, bound)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out1 = torch.nn.functional.conv2d(x, self.weights_3x3, self.biases_3x3, 1, 1)
		out2 = torch.nn.functional.conv2d(x, self.weights_5x5, self.biases_5x5, 1, 2)
		return torch.cat([out1, out2], 1)
	
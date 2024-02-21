# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:15:35 2024

@author: Arino Jenynof
"""
from pathlib import Path
from typing import Literal, Optional, Callable
from torchvision.datasets import VisionDataset
from PIL import Image


class SteganalysisDataset(VisionDataset):
	def __init__(
		self,
		root: str,
		source_dataset: Literal["bows2", "bossbase"],
		method: Literal["wow", "s_uniward"],
		payload: Literal[0.2, 0,4],
		transform: Optional[Callable] = None
	):
		super().__init__(root, transform=transform)
		self.source_dataset = source_dataset
		self.method = method
		self.payload = payload
		
		self.classes = [
			f"{self.source_dataset}-cover",
			f"{self.source_dataset}-{self.method}-{self.payload}"
		]
		
		for class_name in self.classes:
			if not (Path(root) / class_name).exists():
				raise FileNotFoundError(f"'{class_name}' class not found in '{root}', dataset is broken")
		
		self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
		
		self.samples = []
		for class_name in self.classes:
			class_idx = self.class_to_idx[class_name]
			class_path = Path(root) / class_name
			self.samples += [(x.resolve(), class_idx) for x in class_path.iterdir()]
		self.targets = [s[1] for s in self.samples]
	
	def __getitem__(self, index: int):
		path, target = self.samples[index]
		
		with open(path, "rb") as f:
			sample = Image.open(f)
			sample.load()
		
		if self.transform is not None:
			sample = self.transform(sample)
		
		return sample, target
	
	def __len__(self):
		return len(self.samples)

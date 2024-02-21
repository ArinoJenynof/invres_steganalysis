# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:36:43 2024

@author: Arino Jenynof
"""
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import v2
import lightning as L
from tqdm import tqdm
from .dataset import SteganalysisDataset
from .invres_model import LitInvResModel

def main_cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_path")
	args = parser.parse_args()
	
	dataset_path = Path(args.dataset_path).resolve()
	
	model = LitInvResModel()
	
	if not (dataset_path / "train_mean_std.pth").exists():
		_calculate_mean_std(dataset_path)

	train_stats = torch.load(dataset_path / "train_mean_std.pth")
	valid_stats = torch.load(dataset_path / "valid_mean_std.pth")
	test_stats = torch.load(dataset_path / "test_mean_std.pth")
	
	train_transform = v2.Compose([
		v2.ToImage(),
		v2.RandomHorizontalFlip(),
		v2.RandomVerticalFlip(),
		v2.ToDtype(torch.float32, True),
		v2.Normalize([train_stats["mean"]], [train_stats["std"]])
	])
	
	valid_transform = v2.Compose([
		v2.ToImage(),
		v2.ToDtype(torch.float32, True),
		v2.Normalize([valid_stats["mean"]], [valid_stats["std"]])
	])
	
	test_transform = v2.Compose([
		v2.ToImage(),
		v2.ToDtype(torch.float32, True),
		v2.Normalize([test_stats["mean"]], [test_stats["std"]])
	])
	
	bossbase = SteganalysisDataset(str(dataset_path), "bossbase", "wow", 0.4, train_transform)
	bossbase2 = SteganalysisDataset(str(dataset_path), "bossbase", "wow", 0.4, valid_transform)
	bossbase3 = SteganalysisDataset(str(dataset_path), "bossbase", "wow", 0.4, test_transform)
	bows2 = SteganalysisDataset(str(dataset_path), "bows2", "wow", 0.4, train_transform)
	
	indices = list(range(len(bossbase)))
	X_train, X_test, idx_train, idx_test = train_test_split(bossbase.samples, indices, stratify=bossbase.targets, test_size=.5)
	
	X_train, X_valid, idx_train, idx_valid = train_test_split(X_train, idx_train, stratify=[X[1] for X in X_train], test_size=.1)
	
	bossbase_train = torch.utils.data.Subset(bossbase, idx_train)
	bossbase_valid = torch.utils.data.Subset(bossbase2, idx_valid)
	bossbase_test = torch.utils.data.Subset(bossbase3, idx_test)
	
	train_dataset = torch.utils.data.ConcatDataset([bows2, bossbase_train])
	
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 16, True, num_workers=2, persistent_workers=True)
	valid_dataloader = torch.utils.data.DataLoader(bossbase_valid, 16, False, num_workers=2, persistent_workers=True)
	test_dataloader = torch.utils.data.DataLoader(bossbase_test, 16, False, num_workers=2, persistent_workers=True)
	
	trainer = L.Trainer(limit_train_batches=100, max_epochs=10)
	trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
	trainer.test(model, test_dataloader)

def _calculate_mean_std(dataset_path: Path):
	transform = v2.Compose([
		v2.ToImage(),
		v2.ToDtype(torch.float32, True)
	])
	bossbase = SteganalysisDataset(str(dataset_path), "bossbase", "wow", 0.4, transform)
	bows2 = SteganalysisDataset(str(dataset_path), "bows2", "wow", 0.4, transform)
	
	# Original split
	indices = list(range(len(bossbase)))
	X_train, X_test, idx_train, idx_test = train_test_split(bossbase.samples, indices, stratify=bossbase.targets, test_size=.5)
	
	X_train, X_valid, idx_train, idx_valid = train_test_split(X_train, idx_train, stratify=[X[1] for X in X_train], test_size=.2)
	
	bossbase_train = torch.utils.data.Subset(bossbase, idx_train)
	bossbase_valid = torch.utils.data.Subset(bossbase, idx_valid)
	bossbase_test = torch.utils.data.Subset(bossbase, idx_test)
	
	train_dataset = torch.utils.data.ConcatDataset([bows2, bossbase_train])
	
	# Calculate mean and stddev
	train_dataloader = torch.utils.data.DataLoader(train_dataset)
	valid_dataloader = torch.utils.data.DataLoader(bossbase_valid)
	test_dataloader = torch.utils.data.DataLoader(bossbase_test)
	
	mean = torch.zeros(1)
	var = torch.zeros(1)
	for image, _ in tqdm(train_dataloader):
		mean += torch.mean(image)
		var += torch.var(image)
	mean /= len(train_dataset)
	var /= len(train_dataset)
	std = torch.sqrt(var)
	torch.save({"mean": mean, "std": std}, dataset_path / "train_mean_std.pth")
	
	mean = torch.zeros(1)
	var = torch.zeros(1)
	for image, _ in tqdm(valid_dataloader):
		mean += torch.mean(image)
		var += torch.var(image)
	mean /= len(bossbase_valid)
	var /= len(bossbase_valid)
	std = torch.sqrt(var)
	torch.save({"mean": mean, "std": std}, dataset_path / "valid_mean_std.pth")
	
	mean = torch.zeros(1)
	var = torch.zeros(1)
	for image, _ in tqdm(test_dataloader):
		mean += torch.mean(image)
		var += torch.var(image)
	mean /= len(bossbase_test)
	var /= len(bossbase_test)
	std = torch.sqrt(var)
	torch.save({"mean": mean, "std": std}, dataset_path / "test_mean_std.pth")

import torch.nn as nn
from torchvision import models


def get_binary_swin_t():
	"""
	Loads pre-trained Swin-T, freezes its feature extractor,
	and replaces the classification for binary classification.
	"""

	# Load pretrained model
	# 'DEFAULT' pulls the recommended ImageNet weights
	# Roughly 28M params
	model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

	# Freeze all backbone parameters
	for param in model.parameters():
		param.requires_grad = False

	# Replace classifier head to output 2 classes (0: Cat, 1: Dog)
	num_ftrs = model.head.in_features
	model.head = nn.Linear(in_features=num_ftrs, out_features=2)

	return model


# Testing block
if __name__ == "__main__":
	# Instantiate model
	binary_model = get_binary_swin_t()

	# Sanity check
	print("\nMODEL SANITY CHECK:\n")
	print(f"Final layer structure: {binary_model.head}")

	# Check which parameters are trainable
	trainable_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
	total_params = sum(p.numel() for p in binary_model.parameters())

	print(f"Total parameters in network: {total_params}")
	print(f"Trainable parameters: {trainable_params}")

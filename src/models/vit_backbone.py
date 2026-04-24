import torch.nn as nn
from torchvision import models


def get_swin_t(num_classes: int):
	"""
	Loads pre-trained Swin-T, freezes its feature extractor,
	and replaces the classification head.
	"""
	# Load pretrained model
	# 'DEFAULT' pulls the recommended ImageNet weights
	# Roughly 28M params
	model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

	# Freeze all backbone parameters
	for param in model.parameters():
		param.requires_grad = False

	# Replace classifier head
	num_ftrs = model.head.in_features
	model.head = nn.Linear(in_features=num_ftrs, out_features=num_classes)

	return model


# Testing block
if __name__ == "__main__":
	# Instantiate model
	binary_model = get_swin_t(num_classes=2)

	# Sanity check
	print("\nMODEL SANITY CHECK:\n")
	print(f"Final layer structure: {binary_model.head}")

	# Check which parameters are trainable
	trainable_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
	total_params = sum(p.numel() for p in binary_model.parameters())

	print(f"Total parameters in network: {total_params}")
	print(f"Trainable parameters: {trainable_params}")

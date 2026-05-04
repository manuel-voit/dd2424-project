import torch.nn as nn
from torchvision import models


def get_swin(num_classes: int, model_name: str = "swin_t"):
	"""
	Loads pre-trained Swin, freezes its feature extractor,
	and replaces the classification head.
	Available options: swin_t, swin_s, swin_b, swin_v2_t.
	"""
	# Load pretrained model
	if model_name == "swin_t":
		model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT) # ~28.3M params
	elif model_name == "swin_s":
		model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT) # ~49.6M params
	elif model_name == "swin_b":
		model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT) # ~87.8M params
	elif model_name == "swin_v2_t":
		model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT) # ~28.4M params
	else:
		raise ValueError(f"Unsupported Swin model: {model_name}")

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
	binary_model = get_swin(num_classes=2)

	# Sanity check
	print("\nMODEL SANITY CHECK:\n")
	print(f"Final layer structure: {binary_model.head}")

	# Check which parameters are trainable
	trainable_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
	total_params = sum(p.numel() for p in binary_model.parameters())

	print(f"Total parameters in network: {total_params}")
	print(f"Trainable parameters: {trainable_params}")

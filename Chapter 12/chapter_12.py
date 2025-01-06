# -*- coding: utf-8 -*-
"""Chapter 12.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B1UiKy29zsPH5Fu9ia1YbXY74vFVK-k4

# Virtual Environment Setup
"""

# Install virtualenv to create a virtual environment
!pip install virtualenv

# Create a virtual environment called pytorch_env
!virtualenv pytorch_env

# Activate Virtual Environemnt
!source pytorch_env/bin/activate

"""# Installing PyTorch in the virutal environemnt"""

!pip install torch torchvision torchaudio

"""# Code Examples from Chapter 12 of the Book"""

# Example: Post-Training Quantization in PyTorch

import torch
import torch.quantization
from torchvision import models


# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()


# Fuse the model layers (required before quantization)
model.fuse_model()


# Apply post-training static quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)


# Now the model is quantized and ready for deployment

# Example: Applying Unstructured Pruning in PyTorch

import torch
import torch.nn.utils.prune as prune
from torchvision import models


# Load a pre-trained model
model = models.resnet18(pretrained=True)


# Apply global unstructured pruning to all Conv2d layers
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.global_unstructured(
            module, name="weight", amount=0.2
        )


# Remove the pruning reparameterization to make the model ready for deployment
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')
# The pruned model is now ready for deployment

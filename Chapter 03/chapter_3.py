# -*- coding: utf-8 -*-
"""Chapter 3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jZg2TmZoZMn3OZnmGwfi8piI2tgVRc5w

# Virtual Environment Setup
"""

# Install virtualenv to create a virtualz environment
!pip install virtualenv

# Create a virtual environment called pytorch_env
!virtualenv pytorch_env

# Activate Virtual Environemnt
!source pytorch_env/bin/activate

"""# Installing PyTorch in the virutal environemnt"""

!pip install torch torchvision torchaudio

"""# Code Examples from Chapter 3 of the Book"""

# Example Code for torch.compile
import torch


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + x


model = SimpleModel()
compiled_model = torch.compile(model)
compiled_model

# Example showing the simplicity of torch.compile
import torch


# Define a simple model
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2


# Create an instance of the model
model = SimpleModel()


# Compile the model with torch.compile
compiled_model = torch.compile(model)
compiled_model

# PyTorch 1.x example

import torch
import torch.nn as nn


class SimpleModel1x(nn.Module):
    def __init__(self):
        super(SimpleModel1x, self).__init__()
        self.fc = nn.Linear(10, 1)  # Defining a simple linear layer


    def forward(self, x):
        return self.fc(x)


# Create an instance of the model and print it
model1x = SimpleModel1x()
print(model1x)

# PyTorh 2.x Example

import torch
import torch.nn as nn


class SimpleModel20(nn.Module):
    def __init__(self):
        super().__init__()  # Simplified super function call
        self.fc = nn.Linear(10, 1)  # Defining a simple linear layer


    def forward(self, x):
        return self.fc(x)


# Utilizing torch.compile for better performance
compiled_model = torch.compile(SimpleModel20())


# Print the compiled model
print(compiled_model)

# Code Migration: Utilizing torch.compile for better performance

import torch
import torch.nn as nn


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)  # Defining a simple linear layer


    def forward(self, x):
        return self.fc(x)


# Create an instance of the model
model = SimpleModel()


# Now, let's compile the model using torch.compile
compiled_model = torch.compile(
    model,
    fullgraph=False,
    dynamic=None,
    backend='inductor',
    options={
        'epilogue_fusion': True,
        'max_autotune': True,
        'fallback_random': False,
        'shape_padding': True,
        'triton.cudagraphs': True,
        'trace.enabled': True,
        'trace.graph_diagram': True
    }
)
# Print the compiled model to verify
print(compiled_model)

# Code Migration: Updating a Classification Model from PyTorch 1.x to 2.0

import torch
import torch.nn as nn
import torchvision.models as models


# Load the pre-trained MobileNetV3 model
mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)


# Define the classification task
class ClassificationTask20(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, x):
        return self.model(x)


# Utilizing torch.compile for better performance
compiled_task = torch.compile(ClassificationTask20(mobilenet_v3))


# Print the compiled task
print(compiled_task)

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Load the pre-trained DeepLabV3 model with a MobileNetV3-Large backbone
deeplabv3_mobilenet = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)


# Define the segmentation task
class SegmentationTask20(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, x):
        return self.model(x)


# Utilizing torch.compile for better performance
compiled_task = torch.compile(SegmentationTask20(deeplabv3_mobilenet))


# Print the compiled task
print(compiled_task)


# Load and preprocess an image for inference
image = Image.open("example.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)


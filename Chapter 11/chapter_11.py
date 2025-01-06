# -*- coding: utf-8 -*-
"""Chapter 11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xqY8EFyZqkNsFZn3m7Qs3VfZrTi82P8a

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

"""# Code Examples from Chapter 11 of the Book"""

# Standard PyTorch Training Loop:
import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)


    def forward(self, x):
        return self.fc(x)


# Training loop in PyTorch
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


for epoch in range(100):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# PyTorch Lightning Version:
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
class SimpleLightningModel(pl.LightningModule):
    def __init__(self):
        super(SimpleLightningModel, self).__init__()
        self.fc = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()


    def forward(self, x):
        return self.fc(x)


    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# Training with PyTorch Lightning
model = SimpleLightningModel()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_dataloader)

# Converting a PyTorch CNN Model for MNIST
# Standard PyTorch Model and Training Loop:
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# Define a simple CNN for MNIST
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Initialize model, optimizer, and loss function
model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Training loop
for epoch in range(5):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Converting to PyTorch Lightning:
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# Define the CNN model using LightningModule
class LightningCNNModel(pl.LightningModule):
    def __init__(self):
        super(LightningCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


    # Training step defines a single iteration over the batch
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss


    # Validation step (if needed) for performance evaluation
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss


    # Optimizer configuration
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# Initialize the model and Lightning trainer
model = LightningCNNModel()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_dataloader)

# Scripting example
import torch
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output

my_module = MyModule(10,20)
sm = torch.jit.script(my_module)

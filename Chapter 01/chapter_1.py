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

"""# Code Examples from Chapter 1 of the Book"""

# PyTorch dynamic graph example
import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)  # tensor([[1., 1.], [1., 1.]], requires_grad=True)

y = x + 2
print(y)  # tensor([[3., 3.], [3., 3.]], grad_fn=<AddBackward0>)

# The Autograd system example

# Compute forward pass
z = y * y * 3
out = z.mean()
print(z, out)  # tensor([[27., 27.], [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)

# Compute backward pass
out.backward()
print(x.grad)  # tensor([[4.5000, 4.5000], [4.5000, 4.5000]])

# GPU Acceleration example

# Specify device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a tensor and move it to GPU
x = torch.ones([3, 3], device=device)

# Perform operations on the GPU
y = x + 2

# Introduction to TorchScript

import torch
import torchvision.models as models

# An instance of your model.
model = models.resnet50()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

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


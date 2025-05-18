import torch
import torchvision
from torch import nn, optim

from torchvision import datasets, transforms, models

print(f"✅ Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

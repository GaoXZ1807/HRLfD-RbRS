import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

x=torch.randn(200,2)
print(x)
print("--------------")
for i in range(3):
    print(x[i])
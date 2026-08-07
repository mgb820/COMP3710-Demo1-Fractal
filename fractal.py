import torch
import numpy as np
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

points = [[2, 2],[-2, 2],[-2,-2]]

# load into PyTorch tensors
p = torch.Tensor(points)

# transfer to the GPU device
p = p.to(device)

start = torch.Tensor([0.4,0.6])
start = start.to(device)

list = []
rand = random.randint(0,3)
next = (p[rand]+start)/2

for i in range(10000):
    rand = random.randint(0,2)
    next = (p[rand]+next)/2
    list.append(next)
    

# convert back to coordinates
x = [x.cpu().numpy().tolist()[0] for x in list]
y = [y.cpu().numpy().tolist()[1] for y in list]

# plot
import matplotlib.pyplot as plt

plt.scatter(x,y)

plt.tight_layout()
plt.show()
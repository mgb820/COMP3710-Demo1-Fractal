import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

points = [[0, 2],[-1, -1],[1,-1]]

# load into PyTorch tensors
p = torch.Tensor(points)

# transfer to the GPU device
p = p.to(device)

# convert back to coordinates
x = [x[0] for x in p.cpu().numpy().tolist()]
y = [y[1] for y in p.cpu().numpy().tolist()]
print(x)
print(y)

# plot
import matplotlib.pyplot as plt

plt.scatter(x,y)

plt.tight_layout()
plt.show()
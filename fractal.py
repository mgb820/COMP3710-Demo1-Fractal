"""import torch
import numpy as np
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

points = [[0, 2],[-1, -1],[1,-1]]

# load into PyTorch tensors
p = torch.Tensor(points)

# transfer to the GPU device
p = p.to(device)

start = torch.Tensor([0.4,0.6])
start = start.to(device)

list = []
rand = random.randint(0,2)
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

plt.scatter(x,y,s=0.01)

plt.tight_layout()
plt.show()"""

import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a line of points on the complex plane
Y, X = np.mgrid[-0.005:0.005:0.01, 0:1:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)
complex1 = torch.complex(torch.Tensor([1]), torch.Tensor([1]))
complex2 = torch.complex(torch.Tensor([1]),torch.Tensor([-1]))

# transfer to the GPU device
z = z.to(device)
complex1 = complex1.to(device)
complex2 = complex2.to(device)

# Dragon curve
for i in range(12):
    z1 = complex1*z/2
    z2 = 1 - complex2*z/2
    z = torch.cat((z1,z2))

# plot
import matplotlib.pyplot as plt

z_process = z.detach().cpu()
x = z_process.real.numpy()
y = z_process.imag.numpy()

plt.scatter(x, y, s = 0.001)
plt.axis('equal')
plt.tight_layout()
plt.show()
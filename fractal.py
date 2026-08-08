import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a line of points on the complex plane (100 points)
Y, X = np.mgrid[-0.005:0.005:0.01, 0:1:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)

# complex numbers used for iterating (iterating function system)
complex1 = torch.complex(torch.Tensor([1]), torch.Tensor([1]))
complex2 = torch.complex(torch.Tensor([1]),torch.Tensor([-1]))

# transfer to the GPU device
z = z.to(device)
complex1 = complex1.to(device)
complex2 = complex2.to(device)

# Dragon curve for n iterations
for i in range(12):
    z1 = complex1*z/2
    z2 = 1 - complex2*z/2
    z = torch.cat((z1,z2))

# plot
import matplotlib.pyplot as plt

z_process = z.detach().cpu()
x = z_process.real.numpy()
y = z_process.imag.numpy()

# make it look nice
plt.scatter(x, y, s = 0.001)
plt.axis('equal')
plt.tight_layout()
plt.show()
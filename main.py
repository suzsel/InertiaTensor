import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import tplquad

np.set_printoptions(precision=8)

# opens as x, z, y
coords = np.loadtxt('data/djanibekov.txt', unpack=True, delimiter=',', dtype=int)
coords[[1, 2]] = coords[[2, 1]]
coords = coords/max(coords.ravel())
x, y, z = coords

# Coordinate system used for moment of inertia calculation must have its origin at the pivot point
# In case of the top pivot point is the tip of it
x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
z_max = max(z)
p0 = (x_mean, y_mean, z_mean)
# p0 = (x_mean, y_mean, z_max)   # for spinning top
coords = coords.T - p0
coords = coords.T
x, y, z = coords

# plot figure
# using reduced_coords only for efficient plotting
reduced_coords = coords.T[::60].T
x_r, y_r, z_r = reduced_coords
t = np.linspace(0, 1, reduced_coords.shape[1])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_r, y_r, z_r, c=t, cmap='Spectral', linewidths=6, alpha=0.3)
ax.view_init(elev=20)

# calculate Inertia Tensor
N = coords.shape[1]
Ixx = np.sum(y**2 + z**2) / N
Iyy = np.sum(x**2 + z**2) / N
Izz = np.sum(x**2 + y**2) / N
Ixy = np.sum(x * y) / N
Iyz = np.sum(y * z) / N
Ixz = np.sum(x * z) / N

I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
I = np.where(np.abs(I) < 1e-5, 0, I)
print(I)

# draw vectors
origin = np.array([0, 0, 0])
v1 = I[:, 0]
v2 = I[:, 1]
v3 = I[:, 2]

ax.quiver(*origin, *v1, color='r', label='Ixx, Ixy, Ixz', arrow_length_ratio=0.1)
ax.quiver(*origin, *v2, color='g', label='Ixy, Iyy, Iyz', arrow_length_ratio=0.1)
ax.quiver(*origin, *v3, color='b', label='Ixz, Iyz, Izz', arrow_length_ratio=0.1)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

plt.show()

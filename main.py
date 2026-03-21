import numpy as np

# Parameters for this miniproject
side_of_square = 1.0
division_constant = 20

fraction_of_side = side_of_square / division_constant


def u_exact(x, y):
    return x**2 - y**2


x = np.linspace(0, side_of_square, division_constant + 1)
y = np.linspace(0, side_of_square, division_constant + 1)

u = np.zeros((division_constant + 1, division_constant + 1))

# Boundary conditions
u[:, 0] = x**2
u[:, -1] = x**2 - 1
u[0, :] = -(y**2)
u[-1, :] = 1 - y**2

print(fraction_of_side, u.shape, u[0, 0], u[-1, 0], u[0, -1], u[-1, -1])

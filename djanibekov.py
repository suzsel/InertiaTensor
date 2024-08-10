import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import imageio
import os

# Setup directory for saving images
if not os.path.exists('animation_frames'):
    os.makedirs('animation_frames')

np.set_printoptions(precision=8)

# Load and prepare data
coords = np.loadtxt('data/djanibekov.txt', unpack=True, delimiter=',', dtype=int)
coords = coords[:3]  # Assuming the first 3 columns represent x, y, z
coords[[1, 2]] = coords[[2, 1]]  # Swap y and z
coords = coords / max(coords.ravel())  # Normalize
x, y, z = coords

# Finding the origin (pivot point)
x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
z_max = max(z)
p0 = (x_mean, y_mean, z_mean)
coords = coords.T - p0
coords = coords.T
coords = coords.T[::300].T
x, y, z = coords

# Inertia tensor calculation
N = coords.shape[1]
Ixx = np.sum(y ** 2 + z ** 2) / N
Iyy = np.sum(x ** 2 + z ** 2) / N
Izz = np.sum(x ** 2 + y ** 2) / N
Ixy = np.sum(x * y) / N
Iyz = np.sum(y * z) / N
Ixz = np.sum(x * z) / N

I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
I = np.where(np.abs(I) < 1e-5, 0, I)
print("Inertia Tensor:\n", I)

# Set the initial angular velocity to cause the washer to spin around an axis passing through the hole
# Choose a non-principal axis for the initial angular velocity
omega_0 = np.array([1.0, 0.1, 0.1])  # Example initial angular velocity


# Euler's equations of motion
def euler_equations(t, omega, I):
    omega_dot = np.linalg.inv(I) @ np.cross(omega, I @ omega)
    return omega_dot


# Integrate the equations of motion
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)
sol = solve_ivp(euler_equations, t_span, omega_0, t_eval=t_eval, args=(I,))
omega_t = sol.y


# Function to update rotation matrix based on angular velocity
def update_orientation(omega, dt):
    angle = np.linalg.norm(omega) * dt
    axis = omega / np.linalg.norm(omega)
    axis_skew = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * axis_skew + (1 - np.cos(angle)) * axis_skew @ axis_skew
    return R


# Initial object orientation (identity matrix)
orientation = np.eye(3)

# Generate and save each frame as an image
for i in range(len(t_eval)):
    omega = omega_t[:, i]
    dt = t_eval[1] - t_eval[0]
    R = update_orientation(omega, dt)
    orientation = R @ orientation  # Update orientation

    # Apply rotation to the coordinates
    rotated_coords = orientation @ coords
    x_rot, y_rot, z_rot = rotated_coords

    # Create the figure for each frame
    fig = go.Figure(data=[go.Scatter3d(x=x_rot, y=y_rot, z=z_rot, mode='markers')],
                    layout=go.Layout(scene=dict(
                        xaxis=dict(range=[-1, 1], autorange=False),
                        yaxis=dict(range=[-1, 1], autorange=False),
                        zaxis=dict(range=[-1, 1], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1))))

    # Update axis labels
    fig.update_layout(scene=dict(
        xaxis_title='X axis',
        yaxis_title='Y axis',
        zaxis_title='Z axis'
    ))

    # Save the frame as an image
    fig.write_image(f"animation_frames/frame_{i:04d}.png")
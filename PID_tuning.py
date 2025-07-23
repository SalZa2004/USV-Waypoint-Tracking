import control as ctrl
import numpy as np

# Define system parameters
K = 1.0  # system gain
alpha = 1.0  # system time constant
desired_poles = [-2.0]  # Place poles at -2 for critically damped behavior

# Create transfer function (simplified)
num = [K]
den = [1, alpha]
system = ctrl.TransferFunction(num, den)

# Use pole placement to calculate the gains
K_p, K_i, K_d = ctrl.pole_placement(system, desired_poles)

print(f"Calculated PID Gains: Kp={K_p}, Ki={K_i}, Kd={K_d}")

import numpy as np

# Given values
g = 9.81 # m/s^2
m = 5 # kg
k = 3 # N/m
b = 0.5 # N s / m

z0 = 0 # initial position of mass
zdot0 = 0 # m/s, initial velocity of mass

ts = 0.01  # integration time step
ns = 1000 # number of time steps

# Limits of force input
F_max = 20  # N
F_min = -20
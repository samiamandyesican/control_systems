import numpy as np

# given parameters
m1 = 0.35 # kg
m2 = 2
ell = 0.5 # m
g = 9.8 # m/s^2

ts = 0.01 # time step

# initial conditions
z0 = 0.
theta0 = 0.
zdot0 = 0.0
thetadot0 = 0.0

# force limites
F_max = 1.
F_min = -1.
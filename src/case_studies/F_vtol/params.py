import numpy as np

ts = 0.01 # time step

# given params
mc = 1 # kg
Jc = 0.0042 # kg m^2
mr = 0.25 # kg
ml = 0.25 # kg
d = 0.3 # m
mu = 0.1 # kg/s
g = 9.81 # m/s^2

# initial conditions
z_v0 = 0.0
h0 = 0.0
theta0 = 0.0

z_vdot0 = 0.1
hdot0 = -0.1
thetadot0 = 0.2

# force limits
F_min = -1.
F_max = 1.
tau_min = -0.05
tau_max = 0.05

# no idea what this means, just got it from vtol params
unmixer = np.array([[1.0, 1.0], [d, -d]])  # [F, tau] = unmixer @ [fr, fl]
mixer = np.linalg.inv(unmixer)  # [fr, fl] = mixer @ [F, tau]
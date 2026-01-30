import numpy as np

# time step
ts = 0.01

# given parameters
g = 9.81 # m/s
ell1 = 0.247 # m
ell2 = -0.039 
ell3x = -0.007
ell3y = -0.007
ell3z = 0.018
ellT = 0.355
d = 0.12
m1 = 0.108862 # kg
J1x = 0.000189 # kg m^2
J1y = 0.001953 # kg m^2
J1z = 0.001894 # kg m^2
m2 = 0.4717 # kg
J2x = 0.00231 # kg m^2
J2y = 0.003274 # kg m^2
J2z = 0.003416 # kg m^2
m3 = 0.1905 # kg
J3x = 0.0002222 # kg m^2
J3y = 0.0001956 # kg m^2
J3z = 0.000027 # kg m^2

# initial conditions
phi0 = 0
theta0 = 0
psi0 = 0
phidot0 = 0
thetadot0 = 0
psidot0 = 0


##### Chapter 4
# mixing matrices (see end of Chapter 4 in lab manual)
# mixing is a UAV term for taking body forces/torques to individual motor forces
unmixer = np.array([[1.0, 1.0], [d, -d]])  # [F, tau] = unmixer @ [fl, fr]
mixer = np.linalg.inv(unmixer)  # [fl, fr] = mixer @ [F, tau]


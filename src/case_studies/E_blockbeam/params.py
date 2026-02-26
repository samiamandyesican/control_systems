import numpy as np

# given parameters
m1 = 0.35 # kg
m2 = 2
ell = 0.5 # m
g = 9.8 # m/s^2

ts = 0.01 # time step

# initial conditions
z0 = ell/2
theta0 = 0.
zdot0 = 0.0
thetadot0 = 0.0

# linearization/equilibrium point
theta_eq = 0.
z_eq = ell/2
F_eq = 1/2 * m2*g + m1*g*z_eq/ell

# transfer function numerator and denominator
tf_inner_num = [1 / ((1/3 * m2 + 1/4 * m1)*ell)]
tf_inner_den = [1, 0, 0]
tf_outer_num = [-g]
tf_outer_den = [1, 0, 0]

# force limites
F_max = 15.0
F_min = -15.0
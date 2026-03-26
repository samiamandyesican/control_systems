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
x_eq = np.array([z_eq, theta_eq, 0, 0])
F_eq = 1/2 * m2*g + m1*g*z_eq/ell
u_eq = np.array([F_eq])

# transfer function numerator and denominator
tf_inner_num = [1 / ((1/3 * m2 + 1/4 * m1)*ell)]
tf_inner_den = [1, 0, 0]
tf_outer_num = [-g]
tf_outer_den = [1, 0, 0]

# force limites
F_max = 15.0
F_min = -15.0

# State space
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, -g, 0, 0],
    [-3*g*m1/(m2*ell**2+3*m1*z_eq**2), 0, 0, 0]
])
B = np.array([[0], [0], [0], [3*ell / (m2*ell**2+3*m1*z_eq**2)]])
Cm = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0]]) # measure z and theta
Cr = np.array([[1, 0, 0, 0]]) # only command z
D = np.array([[0],[0]])
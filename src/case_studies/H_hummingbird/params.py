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
beta = 0.001 # friction coefficient (N m s/rad)
km = g*(m1*ell1 + m2*ell2) / ellT

# initial conditions
phi0 = 0
theta0 = 0
psi0 = 0
phidot0 = 0
thetadot0 = 0
psidot0 = 0

# linearization/equilibrium point
theta_eq = 0.0
phi_eq = 0.0
psi_eq = 0.0
x_eq = np.array([theta_eq, phi_eq, psi_eq, 0, 0, 0])
x_eq_lat = np.array([phi_eq, psi_eq, 0, 0])
x_eq_lon = np.array([theta_eq, 0])

# transfer function numerators and denominators
tf_th_num = [ellT / (J1y+J2y + m1*ell1**2 + m2*ell2**2)]
tf_th_den = [1, 0, 0]

tf_phi_num = [1/J1x]
tf_phi_den = [1, 0, 0]

F_eq = (m1*ell1 + m2*ell2)*g/ellT
b_psi_phi = F_eq * ellT / (J1z + J2z + J3z + m1*ell1**2 + m2*ell2**2 + m3*ell3x**2 + m3*ell3y**2 + m3*ell3z**2)
tf_psi_num = [b_psi_phi]
tf_psi_den = [1, 0, 0]

# force limits
u_l_max = 1.0
u_l_min = 0.0
u_r_max = 1.0
u_r_min = 0.0

# matrices for state space representation
b_theta = ellT / (m1*ell1**2 + m2*ell2**2 + J1y + J2y)
A_lon = np.array([
    [0, 1],
    [0, 0]
])
B_lon = np.array([[0, b_theta]]).T
Cr_lon = np.array([[1, 0]])


J_T = m1*ell1**2 + m2*ell2**2 + J2z + m3*(ell3x**2 + ell3y**2)
A_lat = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [ellT*F_eq/(J_T + J1x), 0, 0, 0]
])
B_lat = np.array([[0, 0, 1/J1x, 0]]).T
Cr_lat = np.array([[0, 1, 0, 0]])



##### Chapter 4
# mixing matrices (see end of Chapter 4 in lab manual)
# mixing is a UAV term for taking body forces/torques to individual motor forces
unmixer = np.array([[1.0, 1.0], [d, -d]])  # [F, tau] = unmixer @ [fl, fr]
mixer = np.linalg.inv(unmixer)  # [fl, fr] = mixer @ [F, tau]


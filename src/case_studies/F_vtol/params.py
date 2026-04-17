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

z_vdot0 = 0.0
hdot0 = 0.0
thetadot0 = 0.0

# mixers
unmixer = np.array([[1.0, 1.0], [d, -d]])  # [F, tau] = unmixer @ [fr, fl]
mixer = np.linalg.inv(unmixer)  # [fr, fl] = mixer @ [F, tau]

# linearization/equilibrium point
theta_eq = 0.0
h_eq = 0.0
z_v_eq = 0.0
tau_eq = 0.0
F_eq = np.cos(theta_eq)*g*(mc + ml + mr)
x_eq = np.array([z_v_eq, h_eq, theta_eq, 0, 0, 0])
u_eq = np.array([F_eq, tau_eq])

# trsnfer function numerators and denominators
# h / F
tf_al_num = [1 / (mc + mr + ml)]
tf_al_den = [1, 0, 0]
# theta / tau
tf_th_num = [1 / (Jc + 2*mr*d**2)]
tf_th_den = [1, 0, 0]
# z / theta
cweepy_guy = (mc+2*mr)*Jc + (2*mc*mr+4*mr**2)*d**2
tf_z_num = [- F_eq * (Jc + 2 * mr*d**2) / cweepy_guy]
tf_z_den = [1, mu*(Jc+2*mr*d**2)/cweepy_guy, 0]

# state space
A = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, -F_eq/(mc+mr+ml), -mu/(mc+mr+ml), 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
B = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1/(mc+mr+ml), 0],
    [0, 1/(Jc+(mr+ml)*d**2)],
])
Cm = np.eye(3, 6)
Cr = np.eye(2, 6)
D = np.zeros((6, 2))

# force limits
fl_min = 0.0
fl_max = 10.0
fr_min = fl_min
fr_max = fl_max
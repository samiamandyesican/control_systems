"""
Rod-Mass System Parameters

Physical system: Point mass on massless rod with nonlinear spring
"""
import numpy as np
import control as cnt

##### Physical Parameters
m = 0.1              # mass of the point mass (kg)
ell = 0.25           # length of the rod (m)
b = 0.1              # damping coefficient (N⋅m⋅s)
g = 9.8              # gravity (m/s²)
k1 = 0.02            # linear spring coefficient
k2 = 0.01            # cubic spring coefficient (nonlinear term)
tau_max = 3.0        # N, force limit max
tau_min = -3.0       # N, force limit min

# TODO: add any additional parameters needed for simulation or control here ...

ts = 0.01 # time step

# initial conditions
theta0 = 0.0 * np.pi/180 # radians
thetadot0 = 0.0 * np.pi/180

# linearization/equilibrium point
theta_eq = 0.0 * np.pi/180 # radians
thetadot_eq = 0.0 * np.pi/180
x_eq = np.array([theta_eq, thetadot_eq])
tau_eq = m*g*np.cos(theta_eq) + k1*theta_eq + k2*theta_eq**3
tau_fl = lambda theta: m*g*np.cos(theta) + k1*theta + k2*theta**3
u_eq = np.array([tau_eq])

# linearized state space matrices (x = [theta, thetadot], u = [tau])
A21 = (m*g*np.sin(theta_eq) - k1 - 3*k2*theta_eq**2) / (m*ell**2)
A22 = -b / (m*ell**2)
A = np.array([
    [0, 1],
    [A21, A22]
])
B = np.array([
    [0],
    [1 / (m*ell**2)]
])
Cr = np.array([[1, 0]])
Cm = np.array([[1, 0]])

# transfer function numerator/denominator (MONIC)
tf_num = [1/(m*ell**2)]
tf_den = [1, b/(m*ell**2), k1/(m*ell**2)]
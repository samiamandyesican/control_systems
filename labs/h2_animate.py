# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, H_hummingbird

# alias for parameters
P = H_hummingbird.params

# initialize signals for generating data
phi_gen = common.SignalGenerator(amplitude=1, frequency=0.1)
theta_gen = common.SignalGenerator(amplitude=2, frequency=0.2)
psi_gen = common.SignalGenerator(amplitude=3, frequency=0.3)

# initalize data storage
# initial conditions
x0 = np.array([P.phi0, P.theta0, P.psi0, P.phidot0, P.thetadot0, P.psidot0])
x_hist = [x0]
u_hist = []

# loop over time
time = np.arange(start=0, stop=20, step=P.ts, dtype=np.float64) # type: ignore
for t in time[1:]:
    # generate fake state and input data
    x = np.zeros_like(x0, dtype=np.float64)
    x[0] = phi_gen.sin(t)
    x[1] = theta_gen.sin(t)
    x[2] = psi_gen.sin(t)
    x[3:] = [phi_gen.sin(t), theta_gen.sin(t), psi_gen.sin(t)]

    u = np.array([theta_gen.sin(t), theta_gen.sin(t)])

    x_hist.append(x)
    u_hist.append(u)

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

viz = H_hummingbird.Visualizer(time, x_hist, u_hist)
viz.animate()
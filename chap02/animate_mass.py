# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, D_mass

# alias for parameters
P = D_mass.params

# initialize signals for generating data
z_gen = common.SignalGenerator(amplitude=2, frequency=0.1)

# initalize data storage
# initial conditions
x0 = np.array([P.z0, P.zdot0])
x_hist = [x0]
u_hist = []

# loop over time
time = np.arange(start=0, stop=20, step=P.ts, dtype=np.float64) # type: ignore
for t in time[1:]:
    # generate fake state and input data
    x = np.zeros_like(x0, dtype=np.float64)
    x[0] = z_gen.sin(t)
    x[1] = 2*0.1*2*np.pi*np.cos(2*0.1*2*np.pi*t)

    u = np.array([z_gen.sin(t)])

    x_hist.append(x)
    u_hist.append(u)

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

viz = D_mass.Visualizer(time, x_hist, u_hist)
viz.animate()
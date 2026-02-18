# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, H_hummingbird

# alias for parameters
P = H_hummingbird.params

# initialize system and input generator
hummingbird = H_hummingbird.Dynamics()
u_l = 0.5
u_r = 0.5

# initalize data storage
# initial conditions
x_hist = [hummingbird.state]
u_hist = []

# loop over time
time = np.arange(start=0, stop=20, step=P.ts, dtype=np.float64) # type: ignore
for t in time[1:]:
    # generate fake state and input data
    u = np.array([u_l, u_r]) # f_l_gen.sin(t), f_r_gen.sin(t)])

    y = hummingbird.update(u)

    # store data for visualization
    x_hist.append(hummingbird.state)
    u_hist.append(u.flatten())

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

viz = H_hummingbird.Visualizer(time, x_hist, u_hist)
viz.animate()
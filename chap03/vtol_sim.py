# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, F_vtol

# alias for parameters
P = F_vtol.params

# initialize signals for generating data
vtol = F_vtol.Dynamics()
force = common.SignalGenerator(amplitude=0.5, frequency=1.0, y_offset=14.715)
torque = common.SignalGenerator(amplitude=0.001, frequency=1.0, y_offset=-0.01)

# initalize data storage
x_hist = [vtol.state]
u_hist = []

# loop over time
time = np.arange(start=0, stop=20, step=P.ts, dtype=np.float64) # type: ignore
for t in time[1:]:
    # generate input signal
    u = P.mixer @ np.array([[force.sin(t)], [torque.sin(t)]]) # [fr, fl] = mixer @ [F, tau]

     # simulate system response
    y = vtol.update(u)

    # store data for visualization
    u_hist.append(u.flatten())
    x_hist.append(vtol.state)

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)
print(u_hist)

viz = F_vtol.Visualizer(time, x_hist, u_hist)
viz.animate()
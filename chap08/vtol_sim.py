import numpy as np
from numpy import ma

from case_studies import common, F_vtol

blockbeam = F_vtol.Dynamics()
controller = F_vtol.ControllerPD()
z_ref = common.SignalGenerator(amplitude=3.0, frequency=0.04, y_offset=5.0)  
h_ref = common.SignalGenerator(amplitude=2.0, frequency=0.03, y_offset=5.0)
theta_ref = None # allows controller inner loop to fill in this value (not necessary)

time, x_hist, u_hist, r_hist, *_ = common.run_simulation(
    blockbeam,
    [z_ref, h_ref, theta_ref],
    controller,
    controller_input="state",
    t_final=60,
    dt=F_vtol.params.ts,
)

viz = F_vtol.Visualizer(time, x_hist, u_hist, r_hist)
viz.plot()
viz.animate(speed_scale=3.0)
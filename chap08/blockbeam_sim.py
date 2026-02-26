import numpy as np
from numpy import ma

from case_studies import common, E_blockbeam
from case_studies.E_blockbeam.params import ell

blockbeam = E_blockbeam.Dynamics()
controller = E_blockbeam.ControllerPD()
theta_ref = None  # allows controller inner loop to fill in this value (not necessary)
z_ref = common.SignalGenerator(amplitude=0.25, frequency=0.05, y_offset=ell/2)  # can change to 0.2 m for part f)

time, x_hist, u_hist, r_hist, *_ = common.run_simulation(
    blockbeam,
    [z_ref, theta_ref],
    controller,
    controller_input="state",
    t_final=100,
    dt=E_blockbeam.params.ts,
)

viz = E_blockbeam.Visualizer(time, x_hist, u_hist, r_hist)
viz.plot()
viz.animate()
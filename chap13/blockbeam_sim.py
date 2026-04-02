import numpy as np

from case_studies import common, E_blockbeam
from case_studies.E_blockbeam.params import ell

blockbeam = E_blockbeam.Dynamics(alpha = 0.0)
controller = E_blockbeam.ControllerSSIO(separate_integrator=False)

theta_ref = None  # allows controller inner loop to fill in this value (not necessary)
z_ref = common.SignalGenerator(amplitude=0.25, frequency=0.05, y_offset=ell/2)  # can change to 0.2 m for part f)
refs = [z_ref, theta_ref]

d_force = np.array([0.5])

time, x_hist, u_hist, r_hist, xhat_hist, d_hist, *_ = common.run_simulation(
    blockbeam,
    refs,
    controller,
    controller_input="measurement",
    input_disturbance=d_force,
    t_final=100,
    dt=E_blockbeam.params.ts
)

viz = E_blockbeam.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist, d_hist)
viz.plot()
viz.animate()
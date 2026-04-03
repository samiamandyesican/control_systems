import numpy as np

from case_studies import common, E_blockbeam
from case_studies.E_blockbeam.params import ell

blockbeam = E_blockbeam.Dynamics(alpha = 0.2)
controller = E_blockbeam.ControllerSSIDO(separate_integrator=False)

# reference signals
theta_ref = None  # allows controller inner loop to fill in this value (not necessary)
z_ref = common.SignalGenerator(amplitude=0.25, frequency=0.05, y_offset=ell/2)  # can change to 0.2 m for part f)
refs = [z_ref, theta_ref]

# output noise
z_noise = common.SignalGenerator(amplitude=0.001)
theta_noise = common.SignalGenerator(amplitude=0.001)

# input disturbance
d_force = np.array([0.5])

# run simulation
time, x_hist, u_hist, r_hist, xhat_hist, d_hist, dhat_hist = common.run_simulation(
    blockbeam,
    refs,
    controller,
    controller_input="measurement",
    input_disturbance=d_force,
    output_noise = [z_noise, theta_noise],
    t_final=100,
    dt=E_blockbeam.params.ts
)

# visualize results
viz = E_blockbeam.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist, d_hist, dhat_hist)
viz.plot()
# viz.animate()
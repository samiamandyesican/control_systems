import numpy as np

from case_studies import common, F_vtol
from case_studies.F_vtol import params as P

blockbeam = F_vtol.Dynamics(alpha=0.0, F_wind=0.0)
controller = F_vtol.ControllerSSIO(separate_integrator=False)

z_ref = common.SignalGenerator(amplitude=3.0, frequency=0.04, y_offset=5.0)  
h_ref = common.SignalGenerator(amplitude=2.0, frequency=0.03, y_offset=5.0)
theta_ref = None # allows controller inner loop to fill in this value (not necessary)
refs = [z_ref, h_ref, theta_ref]

d_force = P.mixer @ np.array([1.0, 0.1]) # 1 N force, 0.1 N*m torque disturbance

time, x_hist, u_hist, r_hist, xhat_hist, d_hist, *_ = common.run_simulation(
    blockbeam,
    refs,
    controller,
    controller_input="measurement",
    input_disturbance = d_force,
    t_final=60,
    dt=F_vtol.params.ts,
)

viz = F_vtol.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist, d_hist)
viz.plot()
viz.animate(speed_scale=3.0)
import numpy as np

from case_studies import common, F_vtol
from case_studies.F_vtol import params as P

vtol = F_vtol.Dynamics(alpha=0.2, wind_dist=1.0, altitude_disturbance=1.0)
controller = F_vtol.ControllerLQRIDO(separate_integrator=False)

z_ref = common.SignalGenerator(amplitude=3.0, frequency=0.04, y_offset=4.0)  
h_ref = common.SignalGenerator(amplitude=2.0, frequency=0.03, y_offset=3.0)
theta_ref = None # allows controller inner loop to fill in this value (not necessary)
refs = [z_ref, h_ref, theta_ref]

# noise
z_noise = common.SignalGenerator(amplitude=0.001)
h_noise = common.SignalGenerator(amplitude=0.001)
theta_noise = common.SignalGenerator(amplitude=0.001)
output_noise = [z_noise, h_noise, theta_noise]

d_force = P.mixer @ np.array([1.0, 0.1]) # 1 N force, 0.1 N*m torque disturbance

time, x_hist, u_hist, r_hist, xhat_hist, d_hist, dhat_hist = common.run_simulation(
    vtol,
    refs,
    controller,
    controller_input="measurement",
    input_disturbance = d_force,
    output_noise=output_noise,
    t_final=60,
    dt=F_vtol.params.ts,
)

viz = F_vtol.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist, d_hist, dhat_hist, convert_to_F_tau=True)
viz.plot()
# viz.animate(speed_scale=3.0)
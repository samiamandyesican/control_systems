import numpy as np
from case_studies import common, H_hummingbird

hummingbird = H_hummingbird.Dynamics(alpha=0.2)
controller = H_hummingbird.ControllerPID()
phi_ref = None 
theta_ref = common.SignalGenerator(amplitude=np.deg2rad(10), frequency=0.1, y_offset=0.0)
# theta_ref = 0.0  
psi_ref = common.SignalGenerator(amplitude=np.deg2rad(10), frequency=0.07, y_offset=0.0)
refs = [phi_ref, theta_ref, psi_ref]
# phi roll, theta pitch, psi yaw

disturbance = np.array([0.05, 0.05])

time, x_hist, u_hist, r_hist, *_ = common.run_simulation(
    hummingbird,
    refs,
    controller,
    controller_input="measurement",
    input_disturbance=disturbance,
    t_final=60,
    dt=H_hummingbird.params.ts,
)



viz = H_hummingbird.Visualizer(time, x_hist, u_hist, r_hist)
viz.plot()
viz.animate(speed_scale=3.0)
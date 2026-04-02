import numpy as np

from case_studies import common, H_hummingbird

hummingbird = H_hummingbird.Dynamics()
controller = H_hummingbird.ControllerLonPD()
phi_ref = None
theta_ref = common.SignalGenerator(amplitude=np.deg2rad(30), frequency=0.1, y_offset=0.0)
# theta_ref = 0.0  
psi_ref = None

time, x_hist, u_hist, r_hist, *_ = common.run_simulation(
    hummingbird,
    [phi_ref, theta_ref, psi_ref],
    controller,
    controller_input="measurement",
    t_final=60,
    dt=H_hummingbird.params.ts,
)

viz = H_hummingbird.Visualizer(time, x_hist, u_hist, r_hist)
viz.plot()
viz.animate(speed_scale=3.0)
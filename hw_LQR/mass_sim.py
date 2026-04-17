import numpy as np
from case_studies import common, D_mass

mass = D_mass.Dynamics(alpha=0.2)
controller = D_mass.ControllerLQRIDO(separate_integrator=False)
z_ref = common.SignalGenerator(amplitude=1.0, frequency=0.05)
d_force = np.array([0.25])
z_noise = common.SignalGenerator(amplitude=0.001)

time, x_hist, u_hist, r_hist, xhat_hist, d_hist, dhat_hist = common.run_simulation(
    mass,
    [z_ref],
    controller,
    controller_input="measurement",
    input_disturbance=d_force,
    output_noise =[z_noise],
    t_final=60,
    dt=D_mass.params.ts
)

viz = D_mass.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist, d_hist, dhat_hist)
viz.plot()
# viz.animate()
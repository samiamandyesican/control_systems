import numpy as np
from case_studies import common, D_mass

mass = D_mass.Dynamics(alpha=0.0)
controller = D_mass.ControllerSSIO(separate_integrator=False)
z_ref = common.SignalGenerator(amplitude=1.0, frequency=0.05)
d_force = np.array([0.25])

time, x_hist, u_hist, r_hist, xhat_hist, d_hist, *_ = common.run_simulation(
    mass,
    [z_ref],
    controller,
    controller_input="measurement",
    input_disturbance=d_force,
    t_final=60,
    dt=D_mass.params.ts
)

viz = D_mass.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist, d_hist)
viz.plot()
viz.animate()
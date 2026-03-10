import numpy as np
from numpy import ma

from case_studies import common, D_mass

mass = D_mass.Dynamics()
controller = D_mass.ControllerPID()
z_ref = common.SignalGenerator(amplitude=1.0, frequency=0.00001)

time, x_hist, u_hist, r_hist, *_ = common.run_simulation(
    mass,
    [z_ref],
    controller,
    controller_input="measurement",
    t_final=20,
    dt=D_mass.params.ts,
    input_disturbance=0.25 # type:ignore
)

viz = D_mass.Visualizer(time, x_hist, u_hist, r_hist)
viz.plot()
viz.animate()
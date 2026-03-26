# 3rd-party
import numpy as np

# local (controlbook)
from . import params as P
from ..common.dynamics_base import DynamicsBase
from .eom_generated import calculate_eom


class VTOLDynamics(DynamicsBase):
    def __init__(self, alpha=0.0, F_wind=0.0):
        super().__init__(
            # Initial state conditions
            state0=np.array([P.z_v0, P.h0, P.theta0, P.z_vdot0, P.hdot0, P.thetadot0]),
            # Input torque limits
            u_max=np.array([P.fr_max, P.fl_max]),
            u_min=np.array([P.fr_min, P.fl_min]),
            # Time step for integration
            dt=P.ts,
        )
        # see params.py/textbook for details on these parameters
        # mc, Jc, mr, ml, d, mu, g
        self.mc = self.randomize_parameter(P.mc, alpha)
        self.Jc = self.randomize_parameter(P.Jc, alpha)
        self.mr = self.randomize_parameter(P.mr, alpha)
        self.ml = self.randomize_parameter(P.ml, alpha)
        self.d = self.randomize_parameter(P.d, alpha)
        self.mu = self.randomize_parameter(P.mu, alpha)
        self.g = P.g
        self.F_wind = F_wind

    def f(self, x, u):
        xdot = calculate_eom(x, u, self.F_wind, self.mc, self.Jc, self.mr, self.ml, self.d, self.mu, self.g)
        return xdot

    def h(self):
        # return the output equations
        # could also use input u if needed
        z = self.state[0]
        h = self.state[1]
        theta = self.state[2]

        y = np.array([z, h, theta])
        return y

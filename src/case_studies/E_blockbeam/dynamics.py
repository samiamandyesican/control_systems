# 3rd-party
import numpy as np

# local (controlbook)
from . import params as P
from ..common.dynamics_base import DynamicsBase
from .eom_generated import calculate_eom


class BlockbeamDynamics(DynamicsBase):
    def __init__(self, alpha=0.0):
        super().__init__(
            # Initial state conditions
            state0=np.array([P.z0, P.theta0, P.zdot0, P.thetadot0]),
            # Input torque limits
            u_max=P.F_max,
            u_min=P.F_min,
            # Time step for integration
            dt=P.ts,
        )
        # see params.py/textbook for details on these parameters
        self.m1 = self.randomize_parameter(P.m1, alpha)
        self.m2 = self.randomize_parameter(P.m2, alpha)
        self.ell = self.randomize_parameter(P.ell, alpha)
        self.g = P.g # doesn't need uncertainty

    def f(self, x, u):
        xdot = calculate_eom(x, u, self.m1, self.m2, self.ell, self.g)
        return xdot

    def h(self):
        # return the output equations
        # could also use input u if needed
        z = self.state[0]
        theta = self.state[1]
        y = np.array([z, theta])
        return y
  
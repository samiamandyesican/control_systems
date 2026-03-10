# 3rd-party
import numpy as np

# local (controlbook)
from . import params as P
from ..common.dynamics_base import DynamicsBase
from .eom_generated import calculate_eom


class MassDynamics(DynamicsBase):
    def __init__(self, alpha=0.2):
        super().__init__(
            # Initial state conditions
            state0=np.array([P.z0, P.zdot0]),
            # Input torque limits
            u_max=P.F_max,
            u_min=P.F_min,
            # Time step for integration
            dt=P.ts,
        )
        # see params.py/textbook for details on these parameters
        self.m = self.randomize_parameter(P.m, alpha)
        self.k = self.randomize_parameter(P.k, alpha)
        self.b = self.randomize_parameter(P.b, alpha)

    def f(self, x, u):
        xdot = calculate_eom(x, u, self.m, self.k, self.b)
        return xdot

    def h(self):
        # return the output equations
        # could also use input u if needed
        z = self.state[0]
        y = np.array([z])
        return y

import numpy as np

from case_studies import common
from case_studies.L_rodmass import params as P


class RodMassControllerEq(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        self.u_eq = P.u_eq

    def update_with_state(self, r, x):
        theta_ref = r[0]
        theta, thetadot = x

        tau = P.tau_eq
        u_unsat = self.u_eq

        if tau > P.tau_max or tau < P.tau_min:
            print(f"Force saturating at {tau:.3f} to {np.clip(tau, P.tau_min, P.tau_max):.3f} N")

        u = self.saturate(u_unsat, u_max=P.tau_max, u_min=P.tau_min)
        return u

if __name__ == "__main__":
    controller = RodMassControllerEq()
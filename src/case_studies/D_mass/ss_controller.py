import numpy as np
import control as ctrl

from case_studies import common
from case_studies.D_mass import params as P

class MassSSController(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = False):
        # tuning parameters
        t_r = 2.0 # 2 second rise time
        zeta = 0.707 # damping ratio for 5% overshoot

        # check controllability
        if np.linalg.matrix_rank(ctrl.ctrb(P.A, P.B)) != P.A.shape[0]:
            raise ValueError("System not controllable")

        # compute gains
        # desired characteristic equation parameters
        # des_CE = np.poly([p1, p2])
        w_n = np.pi / (2*t_r * np.sqrt(1-zeta**2)) # natural frequency
        des_CE = [1, 2*zeta*w_n, w_n**2]
        des_poles = np.roots(des_CE)
        self.K = ctrl.place(P.A, P.B, des_poles)
        self.kr = -1.0 / (P.Cr @ np.linalg.inv(P.A - P.B @ self.K) @ P.B)
        print(f"{des_poles = }")
        print(f"{self.K = }")
        print(f"{self.kr = }")

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.u_eq = P.u_eq

        self.use_feedback_linearization = use_feedback_linearization


    def update_with_state(self, r, x):
        # unpack references and states:
        x_tilde = x - self.x_eq
        r_tilde = r - self.r_eq

        # compute state feedback control
        u_tilde = -self.K @ x_tilde + self.kr @ r_tilde

        if self.use_feedback_linearization:
            print("ERROR: feedback linearization not implemented for mass system." \
            "\n set use_feedback_linearization=False to use equilibrium force instead.")

        u_unsat = u_tilde + self.u_eq
        u = self.saturate(u_unsat, u_max=P.F_max, u_min=P.F_min)

        return u
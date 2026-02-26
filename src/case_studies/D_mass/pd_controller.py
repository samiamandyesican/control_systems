import numpy as np

from case_studies import common
from case_studies.D_mass import params as P

class MassControllerPD(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = False):
        # tuning parameters
        # p1 = -10
        # p2 = -10
        t_r = 1.65 # 2 second rise time
        zeta = 0.707 # damping ratio for 5% overshoot
        w_n = np.pi / (2*t_r * np.sqrt(1-zeta**2)) # natural frequency

        # system parameters
        b0 = P.tf_num[-1]
        a1, a0 = P.tf_den[-2:]

        # desired characteristic equation parameters
        # s^2 + alpha1*s + alpha0 = s^2 + (a1 + b0*kd)s + (a0 + b0*kp)
        # des_CE = np.poly([p1, p2])
        des_CE = [1, 2*zeta*w_n, w_n**2]
        alpha1, alpha0 = des_CE[-2:]

        # find gains
        self.kp = (alpha0 - a0) / b0
        self.kd = (alpha1 - a1) / b0
        print(f"{self.kp = :.2f}, {self.kd = :.3f}")

        self.F_eq = P.F_eq
        self.use_feedback_linearization = use_feedback_linearization

    def update_with_state(self, r, x):
        # unpack references and states:
        z_ref = r[0]
        z, zdot = x

        # z modified PD
        error = z_ref - z
        F_tilde = self.kp * error - self.kd * zdot

        if self.use_feedback_linearization:
            print("ERROR: feedback linearization not implemented for mass system." \
            "\n set use_feedback_linearization=False to use equilibrium force instead.")

        F = F_tilde + self.F_eq

        if F > P.F_max:
            F = P.F_max

        u = np.array([F])
        return u
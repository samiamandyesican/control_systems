import numpy as np

from case_studies import common
from case_studies.D_mass import params as P

class MassControllerPID(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = False):
        # tuning parameters
        t_r = 1.65 # 2 second rise time
        zeta = 0.707 # damping ratio for 5% overshoot
        self.ki = 2.0
        self.windup_factor = 0.0

        # system parameters
        b0 = P.tf_num[-1]
        a1, a0 = P.tf_den[-2:]

        # desired characteristic equation parameters
        # s^2 + alpha1*s + alpha0 = s^2 + (a1 + b0*kd)s + (a0 + b0*kp)
        # des_CE = np.poly([p1, p2])
        w_n = np.pi / (2*t_r * np.sqrt(1-zeta**2)) # natural frequency
        des_CE = [1, 2*zeta*w_n, w_n**2]
        alpha1, alpha0 = des_CE[-2:]

        # find gains
        self.kp = (alpha0 - a0) / b0
        self.kd = (alpha1 - a1) / b0
        print(f"{self.kp = :.2f}, {self.kd = :.3f}")

        self.F_eq = P.F_eq
        self.use_feedback_linearization = use_feedback_linearization

        # variables for dirty derivative
        self.sigma = 0.05
        self.beta = (2 * self.sigma - P.ts) / (2 * self.sigma + P.ts)
        self.zdot_hat = P.zdot0 # estimated initial derivative value, variable to store previous values
        self.z_prev = P.z0 # variable to store previous values

        # variables for integrator
        self.error_prev = 0.0
        self.error_integral = 0.0


    def update_with_measurement(self, r, y):
        # unpack references and states:
        z_ref = r[0]
        z = y[0]

        # dirty derivative for thetadot
        z_diff = (z - self.z_prev) / P.ts
        self.zdot_hat = self.beta * self.zdot_hat + (1-self.beta) * z_diff
        self.z_prev = z

        # compute state from partially estimated state
        xhat = np.array([z, self.zdot_hat])

        # integrate error
        error = z_ref - z
        err_int_with_windup = P.ts * (error + self.error_prev) / 2
        self.error_integral += err_int_with_windup / (1 + self.windup_factor*self.zdot_hat)
        self.error_prev = error

        # z (modified) PD

        F_tilde = self.kp * error - self.kd * self.zdot_hat + self.ki * self.error_integral

        if self.use_feedback_linearization:
            print("ERROR: feedback linearization not implemented for mass system." \
            "\n set use_feedback_linearization=False to use equilibrium force instead.")

        F = F_tilde + self.F_eq

        if F > P.F_max:
            F = P.F_max

        u_unsat = np.array([F])
        u = self.saturate(u_unsat, u_max=P.F_max, u_min=P.F_min)

        return u, xhat
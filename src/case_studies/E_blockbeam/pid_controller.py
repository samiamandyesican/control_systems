import numpy as np

from case_studies import common
from case_studies.E_blockbeam import params as P


class BlockbeamControllerPID(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        # tuning parameters
        tr_theta = 0.15 # s
        zeta_theta = 0.707
        TS = 10 # time separation between inner and outer loop
        tr_z = tr_theta * TS
        zeta_z = 0.707
        self.ki_z = 0.1
        self.windup_factor_z = 0.0
        self.ki_theta = 0.1
        self.windup_factor_theta = 0.0

        # system parameters
        a1_inner, a0_inner = P.tf_inner_den[-2:]
        b0_inner = P.tf_inner_num[-1]
        a1_outer, a0_outer = P.tf_outer_den[-2:]
        b0_outer = P.tf_outer_num[-1]

        print(
            f"a1_inner = {a1_inner:.3f}, a0_inner = {a0_inner:.3f}, b0_inner = {b0_inner:.3f}\n" \
            f"a1_outer = {a1_outer:.3f}, a0_outer = {a0_outer:.3f}, b0_outer = {b0_outer:.3f}"
        )

        # Inner loop
        wn_theta = np.pi / (2*tr_theta*np.sqrt(1-zeta_theta**2))
        alpha0_inner = wn_theta**2
        alpha1_inner = 2 * zeta_theta * wn_theta
        self.kp_theta = (alpha0_inner - a0_inner) / b0_inner
        self.kd_theta = (alpha1_inner - a1_inner) / b0_inner
        print(f"Inner loop (theta): {self.kp_theta = :.3f}, {self.kd_theta = :.3f}")

        # DC gain of inner loop
        DC_gain = (b0_inner * self.kp_theta) / (a0_inner + b0_inner * self.kp_theta)
        print(f"{DC_gain = :.3f}")

        # Outer loop
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        alpha0_outer = wn_z**2
        alpha1_outer = 2 * zeta_z * wn_z
        self.kp_z = (alpha0_outer - a0_outer) / (b0_outer * DC_gain)
        self.kd_z = (alpha1_outer - a1_outer) / (b0_outer * DC_gain)
        print(f"Outer loop (z): {self.kp_z = :.3f}, {self.kd_z = :.3f}")

        self.F_eq = P.F_eq
        self.use_feedback_linearization = use_feedback_linearization

        # variables for dirty derivatives
        self.sigma = 0.05
        self.beta = (2 * self.sigma - P.ts) / (2 * self.sigma + P.ts)
        self.zdot_hat = P.zdot0 # estimated initial derivative value, variable to store previous values
        self.z_prev = P.z0 # variable to store previous values
        self.thetadot_hat = P.thetadot0 
        self.theta_prev = P.theta0

        # variables for integrator
        self.error_z_prev = 0.0
        self.error_int_z = 0.0
        self.error_theta_prev = 0.0
        self.error_int_theta = 0.0

    def update_with_measurement(self, r, y):
        z_ref = r[0]
        z, theta = y

        # dirty derivative for zdot
        z_diff = (z - self.z_prev) / P.ts
        self.zdot_hat = self.beta * self.zdot_hat + (1-self.beta) * z_diff
        self.z_prev = z

        # dirty derivative for thetadot
        theta_diff = (theta - self.theta_prev) / P.ts
        self.thetadot_hat = self.beta * self.thetadot_hat + (1-self.beta) * theta_diff
        self.theta_prev = theta

        # compute state from partially estimated state
        xhat = np.array([z, theta, self.zdot_hat, self.thetadot_hat])

        # outer loop control
        error_z = z_ref - z
        error_int_z_with_windup = P.ts * (error_z + self.error_z_prev) / 2
        self.error_int_z += error_int_z_with_windup / (1 + self.windup_factor_z*self.zdot_hat)
        self.error_z_prev = error_z

        # theta_ref = self.kp_z * error_z - self.kd_z * zdot
        theta_ref_tilde = self.kp_z * error_z - self.kd_z * self.zdot_hat # + self.ki_z * self.error_int_z
        r[1] = theta_ref_tilde  # if you want to visualize the "reference" angle

        # inner loop control
        error_theta_tilde = theta_ref_tilde - theta
        F_tilde = self.kp_theta * error_theta_tilde - self.kd_theta * self.thetadot_hat

        if self.use_feedback_linearization:
            F_fl = 1/2 * P.m2*P.g + P.m1*P.g* z /P.ell
            F = F_tilde + F_fl
        else:
            F = F_tilde + self.F_eq

        u_unsat = np.array([F])

        if F > P.F_max or F < P.F_min:
            print(f"Force saturating at {F:.3f} to {np.clip(F, P.F_min, P.F_max):.3f} N")

        u = self.saturate(u_unsat, u_max=P.F_max, u_min=P.F_min)
        return u, xhat

if __name__ == "__main__":
    controller = BlockbeamControllerPID()
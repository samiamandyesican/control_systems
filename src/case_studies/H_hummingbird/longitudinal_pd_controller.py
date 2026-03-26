import numpy as np

from case_studies import common
from case_studies.H_hummingbird import params as P


class HummingbirdControllerLonPD(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        # tuning parameters
        tr_th = 0.7 # s
        zeta_th = 0.707
        # TS = 10 # time separation between inner and outer loop
        # tr_z = tr_theta * TS
        # zeta_z = 0.707

        # system parameters
        a1_th, a0_th = P.tf_th_den[-2:]
        b0_th = P.tf_th_num[-1]

        print(
            f"a1_th = {a1_th:.3f}, a0_th = {a0_th:.3f}, b0_th = {b0_th:.3f}\n" \
        )

        # desired characteristic equation and gains
        wn_th = np.pi / (2*tr_th*np.sqrt(1 - zeta_th**2))
        alpha0_th = wn_th**2
        alpha1_th = 2* zeta_th * wn_th
        self.kp_th = (alpha0_th - a0_th) / b0_th
        self.kd_th = (alpha1_th - a1_th) / b0_th
        print(f"Altitude gains: {self.kp_th = :.4f}, {self.kd_th = :.4f}")

        self.use_feedback_linearization = use_feedback_linearization

        # variables for dirty derivative
        self.sigma = 0.05
        self.beta = (2 * self.sigma - P.ts) / (2 * self.sigma + P.ts)
        self.thetadot_hat = P.thetadot0 # estimated initial derivative value, variable to store previous values
        self.theta_prev = P.theta0 # variable to store previous values


    def update_with_measurement(self, r, y):
        # state: [z_v, h, theta, z_vdot, hdot, thetadot]
        theta_ref = r[1]
        phi, theta, psi = y

        # dirty derivative for thetadot
        theta_diff = (theta - self.theta_prev) / P.ts
        self.thetadot_hat = self.beta * self.thetadot_hat + (1-self.beta)*theta_diff
        self.theta_prev = theta

        # compute state from partially estimated state
        xhat = np.array([phi, theta, psi, 0, self.thetadot_hat, 0])

        # compute control force
        error_th = theta_ref - theta
        F_tilde = self.kp_th * error_th - self.kd_th * self.thetadot_hat

        if self.use_feedback_linearization:
            F_fl = (P.m1*P.ell1 + P.m2*P.ell2)*P.g/P.ellT * np.cos(theta)
            F = F_fl + F_tilde
        else:
            print("Equilibrium force not implemented, set use_feedback_linearization to True!")
            F = F_tilde 

        tau = 0.0

        u_unsat = P.mixer @ np.array([F, tau]) # [fr, fl] = mixer @ [F, tau]

        u_unsat = np.array([1/(2*P.km)*(F+tau/P.d), 1/(2*P.km)*(F-tau/P.d)]) # ul, ur
        u_max = np.array([P.u_r_max, P.u_l_max])
        u_min = np.array([P.u_r_min, P.u_l_min])
        u = self.saturate(u_unsat, u_max=u_max, u_min=u_min)
        return u, xhat


if __name__ == "__main__":
    controller = HummingbirdControllerLonPD()
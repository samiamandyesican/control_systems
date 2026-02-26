from os import error

import numpy as np

from case_studies import common
from case_studies.F_vtol import params as P


class AltitudeControllerPD(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = False):
        # tuning parameters
        tr_h = 2.0 # s
        zeta_h = 0.707
        # TS = 10 # time separation between inner and outer loop
        # tr_z = tr_theta * TS
        # zeta_z = 0.707

        # system parameters
        a1_h, a0_h = P.tf_al_den[-2:]
        b0_h = P.tf_al_num[-1]

        print(
            f"a1_h = {a1_h:.3f}, a0_h = {a0_h:.3f}, b0_h = {b0_h:.3f}\n" \
        )

        # desired characteristic equation
        wn_h = np.pi / (2*tr_h*np.sqrt(1 - zeta_h**2))
        alpha0_h = wn_h**2
        alpha1_h = 2* zeta_h * wn_h
        self.kp_h = (alpha0_h - a0_h) / b0_h
        self.kd_h = (alpha1_h - a1_h) / b0_h
        print(f"Altitude gains: {self.kp_h = :.4f}, {self.kd_h = :.4f}")


        # # Inner loop
        # wn_theta = np.pi / (2*tr_theta*np.sqrt(1-zeta_theta**2))
        # alpha0_inner = wn_theta**2
        # alpha1_inner = 2 * zeta_theta * wn_theta
        # self.kp_theta = (alpha0_inner - a0_inner) / b0_inner
        # self.kd_theta = (alpha1_inner - a1_inner) / b0_inner
        # print(f"Inner loop (theta): {self.kp_theta = :.3f}, {self.kd_theta = :.3f}")

        # # DC gain of inner loop
        # DC_gain = (b0_inner * self.kp_theta) / (a0_inner + b0_inner * self.kp_theta)
        # print(f"{DC_gain = :.3f}")

        # # Outer loop
        # wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        # alpha0_outer = wn_z**2
        # alpha1_outer = 2 * zeta_z * wn_z
        # self.kp_z = (alpha0_outer - a0_outer) / (b0_outer * DC_gain)
        # self.kd_z = (alpha1_outer - a1_outer) / (b0_outer * DC_gain)
        # print(f"Outer loop (z): {self.kp_z = :.3f}, {self.kd_z = :.3f}")

        self.F_eq = P.F_eq
        self.use_feedback_linearization = use_feedback_linearization

    def update_with_state(self, r, x):
        # state: [z_v, h, theta, z_vdot, hdot, thetadot]
        h_ref = r[1]
        z_v, h, theta, z_vdot, hdot, thetadot = x

        # apply gains to error
        error_h = h_ref - h
        F_tilde = self.kp_h * error_h - self.kd_h * hdot

        if self.use_feedback_linearization:
            print("Feedback linearization not implemente yet, set use_feedback_linearization to false!")
        else:
            F = F_tilde + self.F_eq
            tau = P.tau_eq

        u_unsat = P.mixer @ np.array([F, tau]) # [fr, fl] = mixer @ [F, tau]
        fr, fl = u_unsat
        if fr > P.fr_max or fr < P.fr_min:
            print(f"Force fr saturating at {fr:.3f} to {np.clip(fr, P.fr_min, P.fr_max):.3f} N")
        if fl > P.fl_max or fl < P.fl_min:
            print(f"Force fl saturating at {fl:.3f} to {np.clip(fl, P.fl_min, P.fl_max):.3f} N")

        u_max = np.array([P.fr_max, P.fl_max])
        u_min = np.array([P.fr_min, P.fl_min])
        u = self.saturate(u_unsat, u_max=u_max, u_min=u_min)
        return u

if __name__ == "__main__":
    controller = AltitudeControllerPD()
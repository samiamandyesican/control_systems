from os import error

import numpy as np

from case_studies import common
from case_studies.F_vtol import params as P


class VTOLControllerPD(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = False):
        # tuning parameters
        tr_h = 3.5 # s
        zeta_h = 0.707
        tr_th = 0.5
        zeta_th = 0.707
        TS = 7.0 # time separation between inner and outer loop
        tr_z = tr_th * TS
        zeta_z = 0.707

        # system parameters
        a1_h, a0_h = P.tf_al_den[-2:]
        b0_h = P.tf_al_num[-1]
        a1_th, a0_th = P.tf_th_den[-2:]
        b0_th = P.tf_th_num[-1]
        a1_z, a0_z = P.tf_z_den[-2:]
        b0_z = P.tf_z_num[-1]

        print(
            f"{a1_h = :.3f}, {a0_h = :.3f}, {b0_h = :.3f}\n",
            f"{a1_th = :.3f}, {a0_th = :.3f}, {b0_th = :.3f}\n",
            f"{a1_z = :.3f}, {a0_z = :.3f}, {b0_z = :.3f}\n"
        )

        # desired characteristic equations
        # h
        wn_h = np.pi / (2*tr_h*np.sqrt(1 - zeta_h**2))
        alpha0_h = wn_h**2
        alpha1_h = 2* zeta_h * wn_h
        self.kp_h = (alpha0_h - a0_h) / b0_h
        self.kd_h = (alpha1_h - a1_h) / b0_h
        print(f"Altitude gains: {self.kp_h = :.4f}, {self.kd_h = :.4f}")
        # theta
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2))
        alpha0_th = wn_th**2
        alpha1_th = 2* zeta_th * wn_th
        self.kp_theta = (alpha0_th - a0_th) / b0_th
        self.kd_theta = (alpha1_th - a1_th) / b0_th
        print(f"Inner loop (theta): {self.kp_theta = :.4f}, {self.kd_theta = :.4f}")
        ## DC gain of inner loop (theta)
        DC_gain = b0_th*self.kp_theta / (a0_th + b0_th*self.kp_theta)
        print(f"{DC_gain = :.4f}") 
        # z
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        alpha0_z = wn_z**2
        alpha1_z = 2 * zeta_z * wn_z
        self.kp_z = (alpha0_z - a0_z) / (b0_z * DC_gain)
        self.kd_z = (alpha1_z - a1_z) / (b0_z * DC_gain)
        print(f"Outer loop (z): {self.kp_z = :.4f}, {self.kd_z = :.4f}")

        self.F_eq = P.F_eq
        self.tau_eq = P.tau_eq
        self.use_feedback_linearization = use_feedback_linearization

    def update_with_state(self, r, x):
        # state: [z_v, h, theta, z_vdot, hdot, thetadot]
        z_ref = r[0]
        h_ref = r[1]
        z, h, theta, zdot, hdot, thetadot = x

        # apply gains to error
        ## h -> F_tilde
        error_h = h_ref - h
        F_tilde = self.kp_h * error_h - self.kd_h * hdot

        ## z -> theta_ref
        error_z = z_ref - z
        theta_ref = error_z * self.kp_z - zdot * self.kd_z
        r[2] = theta_ref

        ## theta - tau_tilde
        error_theta = theta_ref - theta
        tau_tilde = error_theta * self.kp_theta - thetadot * self.kd_theta

        if self.use_feedback_linearization:
            print("Feedback linearization not implemente yet, set use_feedback_linearization to false!")
        else:
            F = F_tilde + self.F_eq
            tau = tau_tilde + self.tau_eq

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
    controller = VTOLControllerPD()
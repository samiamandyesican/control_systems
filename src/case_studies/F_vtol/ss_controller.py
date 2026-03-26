import numpy as np
import control as ctrl

from case_studies import common
from case_studies.F_vtol import params as P


class VTOLControllerSS(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = False):
        # tuning parameters
        tr_h = 3.5 # s
        zeta_h = 0.707
        tr_th = 0.5
        zeta_th = 0.707
        TS = 7.0 # time separation between inner and outer loop
        tr_z = tr_th * TS
        zeta_z = 0.707

        # check controllability
        if np.linalg.matrix_rank(ctrl.ctrb(P.A, P.B)) != P.A.shape[0]:
            raise ValueError("System not controllable")

        # compute gains:
        # h
        wn_h = np.pi / (2*tr_h*np.sqrt(1 - zeta_h**2))
        h_CE = [1, 2*zeta_h*wn_h, wn_h**2]
        h_poles = np.roots(h_CE)
        
        # theta
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2))
        theta_CE = [1, 2*zeta_th*wn_th, wn_th**2]
        theta_poles = np.roots(theta_CE)

        # z
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        z_CE = [1, 2*zeta_z*wn_z, wn_z**2]
        z_poles = np.roots(z_CE)

        des_poles = np.hstack([h_poles, theta_poles, z_poles])

        self.K = ctrl.place(P.A, P.B, des_poles)
        self.kr = - np.linalg.inv(P.Cr @ np.linalg.inv(P.A - P.B @ self.K) @ P.B)
        print(f"{des_poles = }")
        print(f"{self.K = }")
        print(f"{self.kr = }")

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.u_eq = P.u_eq

        self.use_feedback_linearization = use_feedback_linearization

    def update_with_state(self, r, x):
        # state: [z_v, h, theta, z_vdot, hdot, thetadot]
        x_tilde = x - self.x_eq
        r_tilde = r[:2] - self.r_eq

        u_tilde = -self.K @ x_tilde + self.kr @ r_tilde
        u_unsat = u_tilde + self.u_eq
        u_unsat = P.mixer @ u_unsat
        
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
    controller = VTOLControllerSS()
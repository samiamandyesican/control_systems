import numpy as np
import control as ctrl

from case_studies import common
from case_studies.E_blockbeam import params as P


class BlockbeamSSController(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        # tuning parameters
        tr_theta = 0.15 # s
        zeta_theta = 0.707
        TS = 10 # time separation between inner and outer loop
        tr_z = tr_theta * TS
        zeta_z = 0.707

        # check controllability
        if np.linalg.matrix_rank(ctrl.ctrb(P.A, P.B)) != P.A.shape[0]:
            raise ValueError("System not controllable")
        
        # compute gains:

        # Inner loop
        wn_theta = np.pi / (2*tr_theta*np.sqrt(1-zeta_theta**2))
        theta_CE = [1, 2 * zeta_theta * wn_theta, wn_theta**2]
        theta_poles = np.roots(theta_CE)

        # Outer loop
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        z_CE = [1, 2*zeta_z*wn_z, wn_z**2]
        z_poles = np.roots(z_CE)

        des_poles = np.hstack([theta_poles, z_poles])

        self.K = ctrl.place(P.A, P.B, des_poles)
        self.kr = -1.0 / (P.Cr @ np.linalg.inv(P.A - P.B @ self.K) @ P.B)

        print("des_poles:", des_poles)
        print("K:", self.K)
        print("kr:", self.kr)

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.u_eq = P.u_eq

        self.use_feedback_linearization = use_feedback_linearization

    def update_with_state(self, r, x):
        ref = r[0]
        x_tilde = x - self.x_eq
        r_tilde = ref - self.r_eq

        # outer loop (modified) PD
        u_tilde = -(self.K @ x_tilde) + (self.kr @ r_tilde)

        if self.use_feedback_linearization:
            u_fl = np.array([1/2 * P.m2*P.g + P.m1*P.g* x[0] /P.ell])
            u_unsat = u_tilde + u_fl
        else:
            u_unsat = u_tilde + self.u_eq

        # if F > P.F_max or F < P.F_min:
        #     print(f"Force saturating at {F:.3f} to {np.clip(F, P.F_min, P.F_max):.3f} N")

        u = self.saturate(u_unsat, u_max=P.F_max, u_min=P.F_min)
        return u

if __name__ == "__main__":
    controller = BlockbeamSSController()
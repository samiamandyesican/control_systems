import numpy as np
import control as ctrl

from case_studies import common
from case_studies.D_mass import params as P

class MassSSIController(common.ControllerBase):
    def __init__(self, separate_integrator=True):
        # tuning parameters
        tr = 1.1 # 2 second rise time
        zeta = 0.707 # damping ratio for 5% overshoot
        integrator_pole = [-1.2]

        # augmented system
        A1 = np.block([[P.A, np.zeros((P.A.shape[0], 1))],
                       [-P.Cr, np.zeros(1)]])
        B1 = np.vstack((P.B, 0))



        # check controllability
        if np.linalg.matrix_rank(ctrl.ctrb(A1, B1)) != A1.shape[0]:
            raise ValueError("System not controllable")

        # compute gains
        # desired characteristic equation parameters
        # des_CE = np.poly([p1, p2])
        w_n = np.pi / (2*tr * np.sqrt(1-zeta**2)) # natural frequency
        des_CE = [1, 2*zeta*w_n, w_n**2]
        des_sys_poles = np.roots(des_CE)
        des_poles = np.hstack((des_sys_poles, integrator_pole))
        self.K1 = ctrl.place(A1, B1, des_poles)
        self.K = self.K1[:, :-1]
        self.ki = self.K1[:, -1]
        # self.kr = -1.0 / (P.Cr @ np.linalg.inv(P.A - P.B @ self.K) @ P.B)
        print(f"{des_poles = }")
        print(f"{self.K1 = }")
        print(f"{self.K = }")
        print(f"{self.ki = }")

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.x1_eq = np.hstack((self.x_eq, 0))
        self.u_eq = P.u_eq

        # integrator variables
        self.error_prev = 0.0
        self.error_integral = 0.0
        self.separate_integrator = separate_integrator


    def update_with_state(self, r, x):
        # unpack references and states:
        x_tilde = x - self.x_eq
        r_tilde = r - self.r_eq

        # integrate error
        error = r - P.Cr @ x
        self.error_integral += P.ts * (error + self.error_prev)/2
        self.error_prev = error

        # compute state feedback control
        x1_tilde = np.hstack((x_tilde, self.error_integral))
        u_tilde = -self.K @ x_tilde - self.ki @ self.error_integral

        # convert back to original variables (feedback linearization)
        u_unsat = u_tilde + self.u_eq
        u = self.saturate(u_unsat, u_max=P.F_max, u_min=P.F_min)

        return u


if __name__ == "__main__":
    a = MassSSIController(separate_integrator=False)
import numpy as np
import control as ctrl

from case_studies import common
from case_studies.D_mass import params as P
from case_studies.control.dist_observer import DistObserver

class MassSSIDOController(common.ControllerBase):
    def __init__(self, separate_integrator=True):
        # tuning parameters
        Q = 70 *np.diag([3.5, 1.7, 10.0])
        R = np.array([[0.009]])

        # augmented system
        A1 = np.block([[P.A, np.zeros((P.A.shape[0], 1))],
                       [-P.Cr, np.zeros(1)]])
        B1 = np.vstack((P.B, 0))

        controlability_rank = np.linalg.matrix_rank(ctrl.ctrb(A1, B1))
        A1_rank = A1.shape[0]
        print(f"Controlability matrix rank: {controlability_rank}, A1 matrix rank: {A1_rank}")

        # check controllability
        if controlability_rank != A1_rank:
            raise ValueError("System not controllable")

        # compute gains
        # desired characteristic equation parameters
        # des_CE = np.poly([p1, p2])
        self.K1, _, eigs = ctrl.lqr(A1, B1, Q, R)
        self.K = self.K1[:, :-1]
        self.ki = self.K1[:, -1]

        print(f"{self.K1 = }")
        print(f"{self.K = }")
        print(f"{self.ki = }")
        print(f"{eigs = }")

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.x1_eq = np.hstack((self.x_eq, 0))
        self.u_eq = P.u_eq

        # integrator variables
        self.error_prev = 0.0
        self.error_integral = 0.0
        self.separate_integrator = separate_integrator

        # Observer design parameters
        TS_obs = 10.0 # time scale separation between controller and observer
        # des_obs_poles = des_sys_poles.real * TS_obs + des_sys_poles.imag / TS_obs * 1j
        des_obs_poles = np.array([-16, -14])
        disturbance_poles = np.array([-0.7])
        des_obs_poles = np.hstack((des_obs_poles, disturbance_poles))
        print(f"{des_obs_poles = }")

        # observer with disturbance estimation
        self.observer = DistObserver(P.A, P.B, P.Cm, des_obs_poles, P.ts, self.x_eq, self.u_eq)
        self.u_prev = np.zeros(P.B.shape[1])

    def update_with_measurement(self, r, y):
        # update observer with measurement
        xhat, dhat = self.observer.update(y, self.u_prev)

        # unpack references and states:
        x_tilde = xhat - self.x_eq
        r_tilde = r - self.r_eq

        # integrate error
        error = r - P.Cr @ xhat
        self.error_integral += P.ts * (error + self.error_prev)/2
        self.error_prev = error

        # compute state feedback control
        x1_tilde = np.hstack((x_tilde, self.error_integral))
        u_tilde = -self.K @ x_tilde - self.ki @ self.error_integral

        # convert back to original variables (feedback linearization)
        u_unsat = u_tilde + self.u_eq - dhat # feedforward disturbance cancellation
        u = self.saturate(u_unsat, u_max=P.F_max, u_min=P.F_min)
        self.u_prev = u

        return u, xhat, dhat


if __name__ == "__main__":
    a = MassSSIDOController(separate_integrator=False)
import numpy as np
import control as ctrl

from case_studies import common
from case_studies.E_blockbeam import params as P
from case_studies.control.dist_observer import DistObserver


class BlockbeamSSIDOController(common.ControllerBase):
    def __init__(self, separate_integrator=False):
        # tuning parameters
        Q = np.diag([9.0, 1.0, 2, 1.5, 70.0]) # [z, theta, zdot, thetadot, integrator]
        R = np.array([[15.0]])

        # augmented system
        A1 = np.block([[P.A, np.zeros((P.A.shape[0], 1))],
                       [-P.Cr, np.zeros((P.Cr.shape[0], 1))]])
        B1 = np.vstack((P.B, 0))

        # check controllability
        if np.linalg.matrix_rank(ctrl.ctrb(A1, B1)) != A1.shape[0]:
            raise ValueError("System not controllable")
        
        # compute gains:
        self.K1, _, eigs = ctrl.lqr(A1, B1, Q, R)
        self.K = self.K1[:, :-1]
        self.ki = self.K1[:, -1]
        # self.kr = -1.0 / (P.Cr @ np.linalg.inv(P.A - P.B @ self.K) @ P.B)

        print("K1:", self.K1)
        print("K:", self.K)
        print("ki:", self.ki)
        print("eigs:", eigs)

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.u_eq = P.u_eq

        # integrator variables
        self.error_prev = 0.0
        self.error_integral = 0.0
        self.separate_integrator = separate_integrator

        # observer design parameters
        TS_obs = 7.0 # time scale separation between controller and observer
        # des_obs_poles = des_poles[:-1] * TS_obs
        # des_obs_poles = np.array([-160, -145, -120, -140])
        # disturbance_poles = np.array([-5.0])
        des_obs_poles = 1.2 * np.array([-73.2817+73.3038j, -73.2817-73.3038j,  -7.3282 +7.3304j,  -7.3282 -7.3304j, -7.+0.j    ])
        # des_obs_poles = np.hstack((des_obs_poles, disturbance_poles))
        
        self.u_fl = lambda x: np.array([1/2 * P.m2*P.g + P.m1*P.g* self.x_eq[0] /P.ell])
        self.u_prev = np.zeros(P.B.shape[1])
        self.observer = DistObserver(P.A, P.B, P.Cm, des_obs_poles, P.ts, self.x_eq, self.u_eq, self.u_fl)

    def update_with_measurement(self, r, y):
        # update observer with measurement
        xhat, dhat = self.observer.update(y, self.u_prev)

        # convert to linearization (tilde) variables
        ref = r[0]
        x_tilde = xhat - self.x_eq
        r_tilde = ref - self.r_eq


        # integrate error
        error = r_tilde - P.Cr @ x_tilde
        self.error_integral += P.ts * (error + self.error_prev) / 2
        self.error_prev = error

        # compute feedback control
        if self.separate_integrator:
            u_tilde = -self.K @ x_tilde - self.ki @ self.error_integral
        else:
            x1_tilde = np.hstack((x_tilde, self.error_integral))
            u_tilde = -self.K1 @ x1_tilde

        # if F > P.F_max or F < P.F_min:
        #     print(f"Force saturating at {F:.3f} to {np.clip(F, P.F_min, P.F_max):.3f} N")
        # u_fl = np.array([1/2 * P.m2*P.g + P.m1*P.g* x[0] /P.ell])
        u_unsat = u_tilde + self.u_fl(xhat) - dhat # feedforward disturbance cancellation
        u = self.saturate(u_unsat, u_max=P.F_max, u_min=P.F_min)
        self.u_prev = u 

        return u, xhat, dhat

if __name__ == "__main__":
    controller = BlockbeamSSIDOController()
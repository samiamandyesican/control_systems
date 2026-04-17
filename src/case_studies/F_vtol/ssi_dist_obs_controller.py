import numpy as np
import control as ctrl

from case_studies import common
from case_studies.F_vtol import params as P
from case_studies.control.dist_observer import DistObserver


class VTOLControllerSSIDO(common.ControllerBase):
    def __init__(self, separate_integrator=True):
        # tuning parameters
        tr_h = 2.3 # s
        zeta_h = 0.707
        tr_th = 0.4
        zeta_th = 0.707
        TS = 7.0 # time separation between inner and outer loop
        tr_z = tr_th * TS
        zeta_z = 0.707
        integrator_poles = [-70.0, -70.0] # z, h

        # augmented system
        A1 = np.block([[P.A, np.zeros((P.A.shape[0], P.Cr.shape[0]))],
                       [-P.Cr, np.zeros((P.Cr.shape[0], P.Cr.shape[0]))]])
        B1 = np.vstack((P.B, np.zeros((P.Cr.shape[0], P.B.shape[1]))))

        # check controllability
        controlability_rank = np.linalg.matrix_rank(ctrl.ctrb(A1, B1))
        A1_rank = A1.shape[0]
        print(f"Controlability matrix rank: {controlability_rank}, A1 matrix rank: {A1_rank}")

        if controlability_rank != A1_rank:
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

        des_poles = np.hstack([h_poles, theta_poles, z_poles, integrator_poles])

        self.K1 = ctrl.place(A1, B1, des_poles)
        self.K = self.K1[:, :-1]
        self.ki = self.K1[:, -1]
        # self.kr = - np.linalg.inv(P.Cr @ np.linalg.inv(P.A - P.B @ self.K) @ P.B)
        print(f"{des_poles = }")
        print(f"{self.K1 = }")
        print(f"{self.K = }")
        print(f"{self.ki = }")

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.u_eq = P.u_eq

        # integrator variables
        self.error_prev = np.array([0.0, 0.0])
        self.error_integral = np.array([0.0, 0.0])
        self.separate_integrator = separate_integrator

        # observer
        TS_obs = 10.0 # time scale separation between controller and observer
        des_obs_poles = np.hstack([h_poles, theta_poles, z_poles]) * TS_obs
        # des_obs_poles = np.array([-7, -6, -80, -85, -8, -9])
        disturbance_poles = np.array([-20.0, -15.0]) # poles for the disturbance dynamics
        des_obs_poles = np.hstack((des_obs_poles, disturbance_poles))
        print(f"{des_obs_poles = }")

        self.observer = DistObserver(P.A, P.B, P.Cm, des_obs_poles, P.ts, self.x_eq, self.u_eq, mixer=P.mixer)
        self.u_prev = self.u_eq

    def update_with_measurement(self, r, y):
        # update observer with measurement
        xhat, dhat = self.observer.update(y, self.u_prev)

        # state: [z_v, h, theta, z_vdot, hdot, thetadot]
        x_tilde = xhat - self.x_eq
        r_tilde = r[:2] - self.r_eq

        # integrate error
        error = r_tilde - P.Cr @ x_tilde
        self.error_integral += P.ts * (error + self.error_prev) / 2

        # compute feedback control
        if self.separate_integrator:
            u_tilde = -self.K @ x_tilde - self.ki @ self.error_integral
        else:
            x1_tilde = np.hstack((x_tilde, self.error_integral))
            u_tilde = -self.K1 @ x1_tilde

        u_unsat = u_tilde + self.u_eq - dhat # feedforward disturbance cancellation
        u_unsat = P.mixer @ u_unsat 
        fr, fl = u_unsat
        # if fr > P.fr_max or fr < P.fr_min:
        #     print(f"Force fr saturating at {fr:.3f} to {np.clip(fr, P.fr_min, P.fr_max):.3f} N")
        # if fl > P.fl_max or fl < P.fl_min:
        #     print(f"Force fl saturating at {fl:.3f} to {np.clip(fl, P.fl_min, P.fl_max):.3f} N")

        u_max = np.array([P.fr_max, P.fl_max])
        u_min = np.array([P.fr_min, P.fl_min])
        u = self.saturate(u_unsat, u_max=u_max, u_min=u_min)
        self.u_prev = P.unmixer @ u

        return u, xhat, dhat

if __name__ == "__main__":
    controller = VTOLControllerSSIDO()
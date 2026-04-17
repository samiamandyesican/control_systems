from re import A

import numpy as np
import control as ctrl
from numpy.random import f

from case_studies import common
from case_studies.D_mass.params import F_eq
from case_studies.H_hummingbird import params as P


class HummingbirdControllerSSI(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        # tuning parameters
        # pitch
        tr_th = 0.6 # s
        zeta_th = 0.707
        self.ki_th = 0.05

        # roll
        tr_phi = 0.10 # s
        zeta_phi = 0.707

        # yaw
        TS = 7 # time separation between inner (roll) and outer (yaw) loop
        tr_psi = tr_phi*TS
        zeta_psi = 0.707
        self.ki_psi = 0.07

        # anti-windup factor
        self.windup_factor = 0.2

        # augmented system
        A1_lon = np.block([
            [P.A_lon, np.zeros((P.A_lon.shape[0], 1))],
            [-P.Cr_lon, np.zeros((P.Cr_lon.shape[0], 1))]
        ])
        B1_lon = np.vstack((P.B_lon, 0))

        # check controllability
        if np.linalg.matrix_rank(ctrl.ctrb(A1_lon, B1_lon)) != A1_lon.shape[0]:
            raise ValueError("Longitudinal system not controllable")
        
        A1_lat = np.block([
            [P.A_lat, np.zeros((P.A_lat.shape[0], 1))],
            [-P.Cr_lat, np.zeros((P.Cr_lat.shape[0], 1))]
        ])
        B1_lat = np.vstack((P.B_lat, 0))

        print(f"{A1_lat =}")
        print(f"{B1_lat =}")

        if np.linalg.matrix_rank(ctrl.ctrb(A1_lat, B1_lat)) != A1_lat.shape[0]:
            raise ValueError("Lateral system not controllable")

        # def get_gains(tf_num, tf_den, tr, zeta):
        #     a1, a0 = tf_den[-2:]
        #     b0 = tf_num[-1]
        #     wn = np.pi / (2*tr*np.sqrt(1 - zeta**2))
        #     alpha0 = wn**2
        #     alpha1 = 2* zeta * wn
        #     kp = (alpha0 - a0) / b0
        #     kd = (alpha1 - a1) / b0
        #     return kp, kd
                
        # self.kp_th, self.kd_th = get_gains(P.tf_th_num, P.tf_th_den, tr_th, zeta_th)
        # self.kp_phi, self.kd_phi = get_gains(P.tf_phi_num, P.tf_phi_den, tr_phi, zeta_phi)
        # self.kp_psi, self.kd_psi = get_gains(P.tf_psi_num, P.tf_psi_den, tr_psi, zeta_psi) # neglect DC gain of inner loop (roll) since it's equal to 1
        # print(f"Pitch gains: {self.kp_th = :.4f}, {self.kd_th = :.4f}")
        # print(f"Roll gains: {self.kp_phi = :.4f}, {self.kd_phi = :.4f}")
        # print(f"Yaw gains: {self.kp_psi = :.4f}, {self.kd_psi = :.4f}")

        def get_poles(tr, zeta):
            wn = np.pi / (2*tr*np.sqrt(1 - zeta**2))
            real = -zeta*wn
            imag = wn*np.sqrt(1-zeta**2)
            p1 = real + 1j*imag
            p2 = real - 1j*imag
            poles = np.array([p1, p2])
            return poles, wn
        
        phi_poles, _ = get_poles(tr_phi, zeta_phi)
        psi_poles, wn_psi = get_poles(tr_psi, zeta_psi)
        lat_integrator_pole = np.array([-wn_psi/2 * 2]) # place integrator pole at half the natural frequency of the outer loop
        des_poles_lat = np.hstack((phi_poles, psi_poles, lat_integrator_pole))
        self.K1_lat = ctrl.place(A1_lat, B1_lat, des_poles_lat)
        self.K_lat = self.K1_lat[:, :-1]
        self.ki_lat = self.K1_lat[:, -1:] # yaw, psi
        print("Lateral desired poles:", des_poles_lat)
        print("K1_lat:", self.K1_lat)

        theta_poles, wn_theta = get_poles(tr_th, zeta_th)
        lon_integrator_pole = np.array([-wn_theta/2 * 2]) # place integrator pole at half the natural frequency of the system
        des_poles_lon = np.hstack((theta_poles, lon_integrator_pole))
        self.K1_lon = ctrl.place(A1_lon, B1_lon, des_poles_lon)
        self.K_lon = self.K1_lon[:, :-1]
        self.ki_lon = self.K1_lon[:, -1:] # pitch, theta
        print("Longitudinal desired poles:", des_poles_lon)
        print("K1_lon:", self.K1_lon)


        # self.use_feedback_linearization = use_feedback_linearization

        # linearization point
        self.x_eq_lat = P.x_eq_lat
        self.x_eq_lon = P.x_eq_lon
        self.r_eq_lat = P.Cr_lat @ self.x_eq_lat
        self.r_eq_lon = P.Cr_lon @ self.x_eq_lon
        self.x1_eq_lat = np.hstack((self.x_eq_lat, 0))
        self.x1_eq_lon = np.hstack((self.x_eq_lon, 0))

        # variables for dirty derivative
        self.sigma = 0.05
        self.beta = (2 * self.sigma - P.ts) / (2 * self.sigma + P.ts)
        
        self.phidot_hat = P.phidot0 # estimated initial derivative value, variable to store previous values
        self.phi_prev = P.phi0 # variable to store previous values
        self.thetadot_hat = P.thetadot0 # estimated initial derivative value, variable to store previous values
        self.theta_prev = P.theta0 # variable to store previous values
        self.psidot_hat = P.psidot0 # estimated initial derivative value, variable to store previous values
        self.psi_prev = P.psi0 # variable to store previous values

        # variables for integrators for theta and psi
        self.error_prev = np.array([0.0, 0.0])
        self.error_integral = np.array([0.0, 0.0])


    def update_with_measurement(self, r, y):
        # state: [phi, theta, psi, phidot, thetadot, psidot], phi roll, theta pitch, psi yaw
        phi_ref, theta_ref, psi_ref = r[0:3]
        phi, theta, psi = y

        # dirty derivatives for phidot, thetadot, psidot
        ## phidot
        phi_diff = (phi - self.phi_prev) / P.ts
        self.phidot_hat = self.beta * self.phidot_hat + (1-self.beta)*phi_diff
        self.phi_prev = phi 

        ## thetadot
        theta_diff = (theta - self.theta_prev) / P.ts
        self.thetadot_hat = self.beta * self.thetadot_hat + (1-self.beta)*theta_diff
        self.theta_prev = theta

        ## psidot
        psi_diff = (psi - self.psi_prev) / P.ts
        self.psidot_hat = self.beta * self.psidot_hat + (1-self.beta)*psi_diff
        self.psi_prev = psi

        # compute state from partially estimated state
        xhat = np.array([phi, theta, psi, self.phidot_hat, self.thetadot_hat, self.psidot_hat])
        xhat_lat = np.array([phi, psi, self.phidot_hat, self.psidot_hat])
        xhat_lon = np.array([theta, self.thetadot_hat])

        # integrate error for theta and psi
        error_th = theta_ref - theta
        error_psi = psi_ref - psi
        error = np.array([error_th, error_psi])
        err_int_with_windup = P.ts * (error + self.error_prev) / 2
        self.error_integral += err_int_with_windup / (1 + self.windup_factor*(abs(self.thetadot_hat)+abs(self.psidot_hat))) # add some mild windup to prevent integrator from growing too large when the system is saturated and not moving
        self.error_prev = error

        # build augmented state for states feedback with integrator
        x1_lat_tilde = np.hstack((xhat_lat, self.error_integral[1])) - self.x1_eq_lat
        x1_lon_tilde = np.hstack((xhat_lon, self.error_integral[0])) - self.x1_eq_lon

        # compute tilde forces
        F_tilde = -self.K1_lon @ x1_lon_tilde
        tau_tilde = -self.K1_lat @ x1_lat_tilde

        # compute control forces/commands
        F = F_tilde + F_eq
        F = F[0] # ensure 1D
        tau = tau_tilde[0] # ensure 1D
        u_unsat = P.mixer @ np.array([F, tau]) # [fr, fl] = mixer @ [F, tau]
        u_unsat = np.array([1/(2*P.km)*(F+tau/P.d), 1/(2*P.km)*(F-tau/P.d)]) # ul, ur
        u_max = np.array([P.u_r_max, P.u_l_max])
        u_min = np.array([P.u_r_min, P.u_l_min])
        u = self.saturate(u_unsat, u_max=u_max, u_min=u_min)
        return u, xhat


if __name__ == "__main__":
    controller = HummingbirdControllerSSI()
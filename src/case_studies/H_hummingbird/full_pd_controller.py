import numpy as np

from case_studies import common
from case_studies.H_hummingbird import params as P


class HummingbirdControllerFullPD(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        # tuning parameters
        # pitch
        tr_th = 0.7 # s
        zeta_th = 0.707

        # roll
        tr_phi = 0.17 # s
        zeta_phi = 0.707

        # yaw
        TS = 10 # time separation between inner (roll) and outer (yaw) loop
        tr_psi = tr_phi*TS
        zeta_psi = 0.707

        def get_gains(tf_num, tf_den, tr, zeta):
            a1, a0 = tf_den[-2:]
            b0 = tf_num[-1]
            wn = np.pi / (2*tr*np.sqrt(1 - zeta**2))
            alpha0 = wn**2
            alpha1 = 2* zeta * wn
            kp = (alpha0 - a0) / b0
            kd = (alpha1 - a1) / b0
            return kp, kd
        
        self.kp_th, self.kd_th = get_gains(P.tf_th_num, P.tf_th_den, tr_th, zeta_th)
        self.kp_phi, self.kd_phi = get_gains(P.tf_phi_num, P.tf_phi_den, tr_phi, zeta_phi)
        self.kp_psi, self.kd_psi = get_gains(P.tf_psi_num, P.tf_psi_den, tr_psi, zeta_psi) # neglect DC gain of inner loop (roll) since it's equal to 1
        print(f"Pitch gains: {self.kp_th = :.4f}, {self.kd_th = :.4f}")
        print(f"Roll gains: {self.kp_phi = :.4f}, {self.kd_phi = :.4f}")
        print(f"Yaw gains: {self.kp_psi = :.4f}, {self.kd_psi = :.4f}")

        self.use_feedback_linearization = use_feedback_linearization

        # variables for dirty derivative
        self.sigma = 0.05
        self.beta = (2 * self.sigma - P.ts) / (2 * self.sigma + P.ts)
        
        self.phidot_hat = P.phidot0 # estimated initial derivative value, variable to store previous values
        self.phi_prev = P.phi0 # variable to store previous values
        self.thetadot_hat = P.thetadot0 # estimated initial derivative value, variable to store previous values
        self.theta_prev = P.theta0 # variable to store previous values
        self.psidot_hat = P.psidot0 # estimated initial derivative value, variable to store previous values
        self.psi_prev = P.psi0 # variable to store previous values


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

        # compute control forces/commands
        error_th = theta_ref - theta
        F_tilde = self.kp_th * error_th - self.kd_th * self.thetadot_hat

        error_psi = psi_ref - psi
        phi_ref = error_psi * self.kp_psi - self.kd_psi * self.psidot_hat
        r[0] = phi_ref

        error_phi = phi_ref - phi
        tau_tilde = error_phi * self.kp_phi - self.kd_phi * self.phidot_hat

        if self.use_feedback_linearization:
            F_fl = (P.m1*P.ell1 + P.m2*P.ell2)*P.g/P.ellT * np.cos(theta)
            F = F_fl + F_tilde

            tau = tau_tilde # for roll, feedback linearization not needed since it's directly actuated by torque
        else:
            print("Equilibrium force not implemented, set use_feedback_linearization to True!")
            F = F_tilde 

        u_unsat = P.mixer @ np.array([F, tau]) # [fr, fl] = mixer @ [F, tau]

        u_unsat = np.array([1/(2*P.km)*(F+tau/P.d), 1/(2*P.km)*(F-tau/P.d)]) # ul, ur
        u_max = np.array([P.u_r_max, P.u_l_max])
        u_min = np.array([P.u_r_min, P.u_l_min])
        u = self.saturate(u_unsat, u_max=u_max, u_min=u_min)
        return u, xhat


if __name__ == "__main__":
    controller = HummingbirdControllerFullPD()
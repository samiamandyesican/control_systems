# 3rd-party
import numpy as np
import control as ctrl

# local (controlbook)
from . import params as P
from ..common import ControllerBase
from case_studies.control.dist_observer import DistObserver


class RodMassSSIDOController(ControllerBase):
    """
    State-space integral control with disturbance observer for rod-mass system.
    """

    def __init__(self, separate_integrator=True):
        """
        Initialize controller and observer with pole placement design.
        """

        # TODO: initialize necessary variables for your controller here ...
        # roots from pd controller
        # roots_pd = np.array([-33.4129+16.1826j, -33.4129-16.1826j])
        # integrator_pole = np.array([-10.0])
        roots_pd = np.array([-25.+5j, -25-5j])
        integrator_pole = np.array([-15.0])

        # augmented system
        A1 = np.block([[P.A, np.zeros((P.A.shape[0], 1))],
                       [-P.Cr, np.zeros(1)]])
        B1 = np.vstack((P.B, 0))

        # check controllability
        if np.linalg.matrix_rank(ctrl.ctrb(A1, B1)) != A1.shape[0]:
            raise ValueError("System not controllable")
        
        des_poles = np.hstack((roots_pd, integrator_pole))
        self.K1 = ctrl.place(A1, B1, des_poles)
        self.K = self.K1[:, :-1]
        self.ki = self.K1[:, -1]

        print(f"{des_poles = }")
        print(f"{self.K1 = }")
        print(f"{self.K = }")
        print(f"{self.ki = }")

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.x1_eq = np.hstack((self.x_eq, 0))
        self.u_eq = P.u_eq

        # # variables for dirty derivative
        # self.sigma = 0.005
        # self.beta = (2 * self.sigma - P.ts) / (2 * self.sigma + P.ts)
        # self.thetadot_hat = P.thetadot0 
        # self.theta_prev = P.theta0

        # variables for integrator
        self.error_prev = 0.0
        self.error_integral = 0.0
        self.separate_integrator = separate_integrator

        # Observer design parameters
        TS_obs = 5.0 # time scale separation between controller and observer
        # des_obs_poles = des_sys_poles.real * TS_obs + des_sys_poles.imag / TS_obs * 1j
        des_obs_poles = roots_pd*TS_obs
        disturbance_poles = np.array([-1.0])
        des_obs_poles = np.hstack((des_obs_poles, disturbance_poles))
        print(f"{des_obs_poles = }")

        # observer with disturbance estimation
        self.observer = DistObserver(P.A, P.B, P.Cm, des_obs_poles, P.ts, self.x_eq, self.u_eq)
        self.u_prev = np.zeros(P.B.shape[1])

    # TODO: implement functions needed for your controller here ...
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
        u = self.saturate(u_unsat, u_max=P.tau_max, u_min=P.tau_min)
        self.u_prev = u

        return u, xhat, dhat

# 3rd-party
import numpy as np
import control as ctrl

# local (controlbook)
from . import params as P
from ..common import ControllerBase

class RodMassSSIController(ControllerBase):
    """
    State-space integral control for rod-mass system. 
    
    Uses:
    - Full state feedback with integral action
    - No observer (this is for Part 4.1-4.2 of practice final)
    """

    def __init__(self, separate_integrator=True):
        """
        Initialize controller with pole placement design.
        
        """
        # roots from pd controller
        # roots_pd = np.array([-33.4129+16.1826j, -33.4129-16.1826j])
        # integrator_pole = np.array([-10.0])
        roots_pd = np.array([-40., -45])
        integrator_pole = np.array([-30.0])

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

    # TODO: implement functions needed for your controller here ...

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
        u_tilde = -self.K @ x_tilde - self.ki[0] * self.error_integral

        # convert back to original variables (feedback linearization)
        u_unsat = u_tilde + self.u_eq
        u = self.saturate(u_unsat, u_max=P.tau_max, u_min=P.tau_min)

        return u


if __name__ == "__main__":
    controller = RodMassSSIController(separate_integrator=False)
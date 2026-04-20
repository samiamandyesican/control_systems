# 3rd-party
import numpy as np

# local (controlbook)
from . import params as P
from ..common import ControllerBase


class RodMassControllerPID(ControllerBase):
    """
    PID controller for the rod-mass system.
    
    Can be used as PD controller (ki=0) or full PID controller.
    """

    def __init__(self, kp=0., kd=0., ki=0., use_feedback_linearization=False, windup_factor=0.0):
        """
        Initialize PID controller with specified gains.
        
        Args:
            kp: Proportional gain
            kd: Derivative gain 
            ki: Integral gain
        """

        # TODO: initialize necessary variables for your controller here ...
        self.ki = ki
        self.windup_factor_theta = windup_factor
        if kp==0.:
            self.kp = 3/20 * 180/np.pi
        else:
            self.kp = kp

        zeta = 0.9

        # system params
        a1, a0 = P.tf_den[-2:]
        b0 = P.tf_num[-1]
        self.tau_eq = P.tau_eq

        # find kd
        wn = np.sqrt(a0 + b0*self.kp)

        if kd == 0.:
            self.kd = (2*zeta*wn - a1)/b0
        else:
            self.kd = kd

        print(f"{wn = }")
        print(f"{self.kp = }")
        print(f"{self.kd = }")   
        print(f"{self.ki = }")   

        if self.ki == 0.:
            roots_pd =  np.roots([1, a1+b0*self.kd, a0+b0*self.kp])
            print(f"{roots_pd = }")

        # variables for dirty derivative
        self.sigma = 0.005
        self.beta = (2 * self.sigma - P.ts) / (2 * self.sigma + P.ts)
        self.thetadot_hat = P.thetadot0 
        self.theta_prev = P.theta0

        # variables for integrator
        self.error_theta_prev = 0.0
        self.error_int_theta = 0.0

        # feedback linearization
        self.use_feedback_linearization = use_feedback_linearization
        self.tau_fl = P.tau_fl

    # TODO: implement functions needed for your controller here ...
    def update_with_measurement(self, r, y):
        theta_ref = r[0]
        theta = y[0]

        # dirty derivative for thetadot
        theta_diff = (theta - self.theta_prev) / P.ts
        self.thetadot_hat = self.beta * self.thetadot_hat + (1-self.beta) * theta_diff
        self.theta_prev = theta

        # compute state from partially estimated state
        xhat = np.array([theta, self.thetadot_hat])

        # calculate stuff
        error_theta = theta_ref - theta
        error_int_theta_with_windup = P.ts * (error_theta + self.error_theta_prev) / 2
        # if abs(error_theta) < self.windup_factor_theta:
        self.error_int_theta += error_int_theta_with_windup / (1 + self.windup_factor_theta*abs(self.thetadot_hat))
        # else:
        #     self.error_int_theta = 0.0
        self.error_theta_prev = error_theta

        # calculate output
        tau_tilde = self.kp * error_theta - self.kd * self.thetadot_hat + self.ki * self.error_int_theta

        if self.use_feedback_linearization:
            tau = tau_tilde + self.tau_fl(theta)
        else:
            tau = tau_tilde + self.tau_eq

        u_unsat = np.array([tau])

        if tau > P.tau_max or tau < P.tau_min:
            print(f"Force saturating at {tau:.3f} to {np.clip(tau, P.tau_min, P.tau_max):.3f} N")

        u = self.saturate(u_unsat, u_max=P.tau_max, u_min=P.tau_min)
        return u, xhat


if __name__ == "__main__":
    controller = RodMassControllerPID()
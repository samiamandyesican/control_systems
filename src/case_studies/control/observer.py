import numpy as np
import control as ctrl

from case_studies.common.numeric_integration import rk4_step

class Observer:
    def __init__(self, A, B, Cm, des_obs_poles, dt, x_eq, u_eq, u_fl=None):
        self.A = A
        self.B = B
        self.Cm = Cm
        self.dt = dt
        self.x_eq = x_eq
        self.u_eq = u_eq
        self.u_fl = u_fl # note this is a function that takes in the observer state and computes the feedforward control for feedback linearization (if applicable)

        # check observability
        observability_rank = np.linalg.matrix_rank(ctrl.ctrb(A.T, Cm.T))
        A_rank = A.shape[0]

        print(f"Observability matrix rank: {observability_rank}, A matrix rank: {A_rank}")

        if observability_rank != A_rank:
            raise ValueError("System not observable")

        # compute observer gains
        self.L = ctrl.place(A.T, Cm.T, des_obs_poles).T
        print(f"Observer poles: {des_obs_poles}")
        print(f"Observer gain matrix L:\n{self.L}")

        # initialize observer variables
        self.xhat = np.zeros(A.shape[0])
        self.u_prev = np.zeros(B.shape[1])

    def observer_f(self, xhat, y):
        # comopute tilde variables
        xhat_tilde = xhat - self.x_eq
        y_error = y - self.Cm @ xhat

        if self.u_fl is not None:
            try:
                u_tilde = self.u_prev - self.u_fl(xhat)
            except Exception as e:
                print(f"Error computing feedforward control: {e}. is self.u_fl a function of xhat?")
                print("Proceeding without feedforward control.")
                self.u_fl = None
        else:
            try:
                u_tilde = self.u_prev - self.u_eq
            except Exception as e:
                print(f"Error computing u_tilde without feedforward control: {e}. is self.u_eq defined?")
                print("Proceeding with u_tilde = self.u_prev.")
                u_tilde = self.u_prev
        
        # compute observer dynamics
        xhat_tilde_dot = self.A @ xhat_tilde + self.B @ u_tilde + self.L @ y_error
        
        return xhat_tilde_dot
    
    def update(self, y, u_prev):
        self.u_prev = u_prev
        xhat = rk4_step(self.observer_f, self.xhat, y, self.dt)
        self.xhat = xhat
        return xhat
import numpy as np
import control as ctrl

from case_studies.common.numeric_integration import rk4_step

class DistObserver:
    def __init__(self, A, B, Cm, des_obs_poles, dt, x_eq, u_eq, u_fl=None, mixer=None):
        self.A = A
        self.B = B
        self.Cm = Cm
        self.dt = dt
        self.x_eq = x_eq
        self.u_eq = u_eq
        self.u_fl = u_fl # note this is a function that takes in the observer state and computes the feedforward control for feedback linearization (if applicable)
        self.mixer = mixer

        # augmented system for disturbance estimation
        num_inputs = B.shape[1] # assume number of disturbances is equal to number of inputs
        self.A2 = np.block([[A, B], [np.zeros((num_inputs, A.shape[1])), np.zeros((num_inputs, num_inputs))]])
        self.B2 = np.block([[B], [np.zeros((num_inputs, num_inputs))]])
        self.Cm2 = np.block([Cm, np.zeros((Cm.shape[0], num_inputs))])


        # check observability
        observability_rank = np.linalg.matrix_rank(ctrl.ctrb(self.A2.T, self.Cm2.T))
        A2_rank = self.A2.shape[0]

        print(f"Observability matrix rank: {observability_rank}, A matrix rank: {A2_rank}")

        if observability_rank != A2_rank:
            raise ValueError("System not observable")

        # compute observer gains
        self.L2 = ctrl.place(self.A2.T, self.Cm2.T, des_obs_poles).T
        print(f"Observer poles: {des_obs_poles}")
        print(f"Observer gain matrix L:\n{self.L2}")

        # initialize observer variables
        self.x2hat_tilde = np.zeros(self.A2.shape[0])
        self.u_prev = np.zeros(num_inputs)
        self.x2_eq = np.hstack((self.x_eq, np.zeros(num_inputs)))

    def observer_f(self, x2hat_tilde, y):
        # compute tilde variables
        x2hat = x2hat_tilde + self.x2_eq
        y_error = y - self.Cm2 @ x2hat

        if self.u_fl is not None:
            try:
                u_tilde = self.u_prev - self.u_fl(x2hat)
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
        x2hat_tilde_dot = self.A2 @ x2hat_tilde + self.B2 @ u_tilde + self.L2 @ y_error
        
        return x2hat_tilde_dot
    
    def update(self, y, u_prev):
        self.u_prev = u_prev
        x2hat_tilde = rk4_step(self.observer_f, self.x2hat_tilde, y, self.dt)
        self.x2hat_tilde = x2hat_tilde
        x2hat = x2hat_tilde + self.x2_eq
        xhat = x2hat[:-self.B.shape[1]]
        dhat = x2hat[-self.B.shape[1]:]
        if self.mixer is not None:
            dhat = self.mixer @ dhat
        
        return xhat, dhat
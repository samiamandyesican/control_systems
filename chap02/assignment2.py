import sympy as sp

def A2E():
    # l = 0.5 # m
    # m1 = 0.35 # kg
    # m2 = 2
    # g = 9.81 # m/s

    # define sympy symbols (dependent and independent)
    t, ell, m_1, m_2, g = sp.symbols("t ell m_1 m_2 g")
    theta_f, z_f = sp.symbols("theta z", cls=sp.Function) # dependent variables
    theta = theta_f(t)
    z = z_f(t)
 
    # define positions and velocities in terms of origin
    p1 = sp.Matrix([[z*sp.cos(theta)], [z*sp.sin(theta)], [0]])
    p1_dot = p1.diff(t)

    p2 = sp.Matrix([[ell/sp.Integer(2) * sp.cos(theta)],
                    [ell/sp.Integer(2) * sp.sin(theta)],
                    [0]])
    p2_dot = p2.diff(t)

    # define angular velocity vector
    w = sp.Matrix([[0],
                   [0],
                   [sp.diff(theta, t)]])
    
    # define rotational inertia tensor
    # ONLY WORKS FOR ROTATION ABOUT khat! Otherwise you need rotation matrices
    J2 = sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, sp.Rational(1, 12) * m_2 * ell**2]
    ])
    K_1 = sp.Rational(1, 2) * m_1 * p1_dot.T * p1_dot
    K_2_lin = sp.Rational(1, 2) * m_2 * p2_dot.T * p2_dot
    K_2_rot =  sp.Rational(1, 2) * w.T * J2 * w

    K = K_1 + K_2_lin + K_2_rot 

    sp.pprint(sp.simplify(K)[0])


def A2F():
    # define symbols (independent, dependent)
    t, J_c, m_c, m_r, m_l, d = sp.symbols("t J_c m_c m_r m_l d")
    z_v_f, h_f, theta_f = sp.symbols("z_v h theta", cls=sp.Function)
    z_v = z_v_f(t)
    h = h_f(t)
    theta = theta_f(t)

    # define positions
    p_c = sp.Matrix([
        [z_v], [h], [0]
    ])

    p_r = p_c + sp.Matrix([
        [d*sp.cos(theta)],
        [d*sp.sin(theta)],
        [0]
    ])

    p_l = p_c + sp.Matrix([
        [-d*sp.cos(theta)],
        [-d*sp.sin(theta)],
        [0]
    ])

    # define velocities
    p_c_dot = p_c.diff(t)
    p_r_dot = p_r.diff(t)
    p_l_dot = p_l.diff(t)

    # angular velocity
    w = sp.Matrix([[0], [0], [sp.diff(theta, t)]])

    KE_c = sp.Rational(1, 2) * m_c * p_c_dot.T * p_c_dot + sp.Rational(1, 2) * w.T * J_c * w
    KE_r = sp.Rational(1, 2) * m_r * p_r_dot.T * p_r_dot
    # KE_l = sp.Rational(1, 2) * m_l * p_l_dot.T * p_l_dot
    KE_l = sp.Rational(1, 2) * m_r * p_l_dot.T * p_l_dot

    KE = KE_c + KE_r + KE_l

    sp.pprint(sp.simplify(KE[0]))


if __name__ == "__main__":
    # This turns on the best available formatting for your environment
    # A2E()
    A2F()
import sympy as sp
from case_studies.common.sym_utils import dynamicsymbols, printeq, DynamicSymbol

# define symbols (independent, dependent)
t, J_c, m_c, m_r, m_l, d, g, mu = sp.symbols("t J_c m_c m_r m_l d g mu")
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
KE = sp.simplify(KE[0, 0])